def overall_success(course,walk):
 n=course['summary']['model'];w=walk['summary']
 # End-to-end attempts remain in the denominator. Walking-prefix failures are
 # attributed separately, but cannot inflate the overall success rate.
 return (n['settled']+w['successful_stops'])/(n['cases']+w['attempted'])

def passes(course,walk):
 assert course['domain_randomization_enabled'] and walk['effective_dr']['enabled']
 assert course['effective_dr']['level']==walk['level']
 return overall_success(course,walk)>=.9

def advance_course(level,accepted):
 assert 0<=level<=3
 if not accepted:return level,False
 return min(level+1,3),level==3
