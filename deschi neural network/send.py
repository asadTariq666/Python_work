import paho.mqtt.client as mqtt
import numpy as np
from PIL import  Image
import json

def on_connect(client, userdata, flags, rc):
    if rc == 0:

        print("Connected. ")
        client.subscribe("Group_23/Image/predict")
    else:
        print("Failed to connect. Error code: %d."% rc)


def on_message(client,userdata,msg):
    print("Received message from Server.")
    resp_dict = json.loads(msg.payload)
    print("Filename:%s, Prediction: %s, Score: %3.4f" % (resp_dict["filename"],resp_dict["prediction"], resp_dict['score']))


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect 
    client.on_message = on_message 
    client.connect(hostname) 
    client.loop_start() 
    return client

def load_image(filename):
    img = Image.open(filename) 
    img = img.resize((249, 249)) 
    imgarray = np.array(img) / 255.0 
    final = np.expand_dims(imgarray, axis = 0) 
    return final

def send_image(client, filename):
    img = load_image(filename) 
    img_list = img.tolist() 
    send_dict = {"filename":filename, "data": img_list} 
    client.publish("Group_99/IMAGE/classify", json.dumps(send_dict))

def main():
    client = setup("192.168.0.1") 
    print("Sending data.")
    send_image(client, "tulips2.jpg") 
    print("Done. Waiting for results.")
    while True:
       pass
if __name__ == '__main__':
        main()