"""Diagnostic path family, never invoked by the PPO environment."""
import numpy as np
from .motor_path import ContinuousMotorPath
from .acceptance import geometric_success


def make_quiet_path(parameters, start, goal, low, high):
    """Three sagittal waypoints, unchanged head targets and physical bounds."""
    p = np.atleast_2d(parameters)
    fractions = np.array([0., .25, .5, .75, 1.])
    nodes = start[None, None] + fractions[None, :, None] * (goal - start)[None, None]
    nodes = np.broadcast_to(nodes, (len(p), 5, 14)).copy()
    offset = p[:, :9].reshape(-1, 3, 3)
    nodes[:, 1:4, 2:5] += offset
    nodes[:, 1:4, 11:14] -= offset
    nodes[:, :, 5:9] = 0.
    nodes = np.clip(nodes, low, high)
    duration = np.clip(p[:, 9], 1., 5.)
    return ContinuousMotorPath(nodes, np.repeat((duration / 4)[:, None], 4, axis=1))


def make_support_transfer_path(parameters, base_nodes, low, high):
    """Alternative feet-to-hip support path, with independent leg corrections.

    Diagnostic only. The endpoint leg pose is allowed to change; the physical
    success check still requires commanded height and geometric uprightness.
    """
    p=np.atleast_2d(parameters); base=np.asarray(base_nodes)
    nodes=np.broadcast_to(base,(len(p),*base.shape)).copy()
    fractions=np.linspace(0.,1.,len(base)); knots=np.linspace(0.,1.,5)
    for trial in range(len(p)):
        shared=np.vstack([np.zeros(3),p[trial,:9].reshape(3,3),p[trial,25:28]])
        asym=np.vstack([np.zeros(3),p[trial,16:25].reshape(3,3),np.zeros(3)])
        for axis in range(3):
            a=np.interp(fractions,knots,shared[:,axis])
            b=np.interp(fractions,knots,asym[:,axis])
            nodes[trial,:,2+axis]+=a+b
            nodes[trial,:,11+axis]+=-a+b
    nodes[:,:,5:9]=0.
    nodes=np.clip(nodes,low,high)
    duration=np.clip(p[:,9],1.,5.)
    return ContinuousMotorPath(nodes,np.repeat((duration/(len(base)-1))[:,None],len(base)-1,axis=1))


def support_transfer_nodes(stand, sit, geometry_rows, case, stepping=False, flat_endpoint=False):
    """Build diagnostic leg nodes from verified FK and bounded foot relocation.

    CAD FK shows left roll negative/right roll positive lift the seated feet,
    while left yaw positive/right yaw negative move them back toward the COM.
    These proposals are not an assertion that the feet lift in loaded dynamics.
    """
    valid=[r for r in geometry_rows if max(abs(v) for v in r['residual'])<1e-5 and r['lowest_z']>=-1e-5]
    if not valid:raise ValueError('No valid upright flat-foot support pose')
    chosen=[valid[j] for j in np.linspace(0,len(valid)-1,7).astype(int)]
    flat=np.repeat(np.asarray(stand)[None],7,axis=0)
    for j,row in enumerate(chosen):
        flat[j,2:5]=row['angles'];flat[j,11:14]=-np.array(row['angles'])
    flat[0]=stand
    if flat_endpoint:
        nodes=flat
    elif not stepping:
        nodes=np.vstack([flat,sit])
    else:
        # Sit-to-stand proposal: unload/reposition feet individually before
        # extending both legs. Retain original seated reset as first node.
        pre=np.repeat(np.asarray(sit)[None],8,axis=0)
        pre[1:,0]=.4;pre[1:4,1]=-.10
        pre[2:,2:5]=flat[-1,2:5]
        pre[4:,9]=-.4;pre[4:7,10]=.18
        pre[5:,11:14]=flat[-1,11:14]
        pre[7]=flat[-1];pre[7,[0,9]]=[.4,-.4]
        tail=flat[-2::-1].copy()
        fade=np.linspace(1.,0.,len(tail));tail[:,0]=.4*fade;tail[:,9]=-.4*fade
        nodes=np.vstack([pre,tail])[::-1].copy()
    return nodes[::-1].copy() if case==2 else nodes


def balance_targets(desired, parameters, up, gyro, low, high):
    """Bounded sagittal feedback used only to probe physical feasibility."""
    p = np.atleast_2d(parameters)
    pitch = np.arctan2(-up[:, 0], up[:, 2])
    target = desired.copy()
    hip = p[:, 10] * pitch + p[:, 12] * gyro[:, 1]
    ankle = p[:, 11] * pitch + p[:, 13] * gyro[:, 1]
    target[:, 2] += hip
    target[:, 11] -= hip
    target[:, 4] += ankle
    target[:, 13] -= ankle
    if p.shape[1] >= 16:
        roll = np.arctan2(up[:, 1], up[:, 2])
        correction = p[:, 14] * roll + p[:, 15] * gyro[:, 0]
        target[:, 1] += correction
        target[:, 10] += correction
    target[:, 5:9] = 0.
    return np.clip(target, low, high)


def support_feedback_parameters(parameters, elapsed):
    """Allow feedback to change while feet and hips exchange support."""
    p=np.atleast_2d(parameters).copy()
    phase=np.clip(np.atleast_1d(elapsed)/np.clip(p[:,9],1.,5.),0.,1.)
    blend=np.clip((phase-.55)/.35,0.,1.)[:,None]
    p[:,10:16]=(1-blend)*p[:,10:16]+blend*p[:,28:34]
    return p


def summarize_quiet_probe(data, goal_height, dt=.02, movement_start=1., body_limit_deg=5., head_limit_deg=5.):
    success, _ = geometric_success(data['height'] - goal_height, data['tilt'], data['head_tilt'], data['base_speed'], data['qvel'], dt)
    fallen = data['fallen'].any(axis=0)
    success &= ~fallen
    start = round(movement_start / dt)
    peak_body = np.rad2deg(data['tilt'][start:].max(axis=0))
    head_deviation = np.rad2deg(np.abs(data['qpos'][start:, :, 12:16]).max(axis=(0, 2)))
    quiet = success & (peak_body <= body_limit_deg) & (head_deviation <= head_limit_deg)
    score = (200 * success - 200 * fallen
             - 80 * np.abs(data['height'][-50:].mean(axis=0) - goal_height) / .015
             - 2 * peak_body - 5 * head_deviation + 100 * quiet)
    return dict(success=success, fallen=fallen, peak_body_deg=peak_body,
                head_deviation_deg=head_deviation, quiet=quiet, score=score)
