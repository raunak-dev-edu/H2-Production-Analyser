import json
FILE='history.json'

def save_session(sess):
    try: data=json.load(open(FILE))
    except: data={'sessions':[]}
    data['sessions'].append(sess)
    json.dump(data,open(FILE,'w'),indent=2)

def get_sessions():
    try: return json.load(open(FILE))['sessions']
    except: return []