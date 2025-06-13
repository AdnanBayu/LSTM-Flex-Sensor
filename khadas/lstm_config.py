import random

BROKER = 'localhost'
PORT = 1883
TOPIC = "glovitoo/sensors"
CLIENT_ID = f'subscribe-{random.randint(0, 100)}'
USERNAME = 'gl_proto'
PASSWORD = 'prototipe123'
