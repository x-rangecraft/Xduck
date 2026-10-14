def overall_success(nominal, challenge):
    n=nominal['summary']['model'];c=challenge['summary']['model']
    return (n['settled']+c['settled'])/(n['cases']+c['cases'])

def passes(nominal, challenge):
    return overall_success(nominal,challenge)>=.9
