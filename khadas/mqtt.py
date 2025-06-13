import random
from paho.mqtt import client as mqtt_client

BROKER = 'localhost'
PORT = 1883
TOPIC = "glovitoo/sensors"
CLIENT_ID = f'subscribe-{random.randint(0, 100)}'
USERNAME = 'gl_proto'
PASSWORD = 'prototipe123'

def get_client(client_id, username, password, broker, port) -> mqtt_client:
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)
    client.username_pw_set(username=username, password=password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def on_message(client, userdata, msg):
    data = msg.payload.decode().split(',')
    data = data[:4] + data[8:-2] + [data[-1].strip()]
    data = [float(x) for x in data]
    print(data)

def subscribe(client: mqtt_client, topic, loop=on_message):
    client.subscribe(topic)
    client.on_message = loop

def run():
    client = get_client(client_id=CLIENT_ID, username=USERNAME, password=PASSWORD, broker=BROKER, port=PORT)
    subscribe(client, topic=TOPIC)
    client.loop_forever()

if __name__ == '__main__':
	run()
