from flask import Flask, request, jsonify
from Physion import *
from getFrames import *
import time
import urllib.request


application = Flask(__name__)

def run_plank(video_URL):
    get_frames(video_URL)
    getPlank()
    return("Very cool and Awesome")
    

@application.route("/endpoint", methods=['POST'])
def PoseEstimation_endpoint():
    print('past def pose estimation')
    if request.method == 'POST':
        print('past POST')
        json_dict = request.get_json(force = True)
        print('past get json')
        print(json_dict)
        if "video_URL" in json_dict:
            print('past get video url')
            result = run_plank(json_dict['video_URL'])
            print(result)
            return jsonify({'output' : result})
        else:
            return jsonify({
                "status": "failed",
                "message": "parameter 'my_text' is required!"
            })
    else:
        pass



if __name__=='__main__':
  application.run()