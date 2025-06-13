#include <WiFi.h>
#include <MQTT.h>
#include <Mux.h>

#include <Adafruit_MPU6050.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_Sensor.h>

const char SSID[] = "khadas_ap";
const char PASS[] = "gluvitu321";
const int WIFI_TIMEOUT = 20000; //ms

#define oled_add 0x3C
#define SET_A 27
#define SET_B 16
#define SET_C 17
#define SEN_IN 34

WiFiClient net;
MQTTClient client;

unsigned long lastMillis = 0, lastSensorSend=0;

Adafruit_MPU6050 mpu;
Adafruit_SSD1306 display = Adafruit_SSD1306(128, 32, &Wire);
admux::Mux glove(admux::Pin(SEN_IN, INPUT, admux::PinType::Analog), admux::Pinset(SET_A, SET_B, SET_C));


String allSensor(){
  sensors_event_t a,g,temp;
  float yaw, pitch;
  mpu.getEvent(&a, &g, &temp);
  String data = "";

  for(byte i=0; i<glove.channelCount();i++){
    data += String(glove.read(i)) + ",";
  }

  data += String(a.acceleration.x) + ",";
  data += String(a.acceleration.y) + ",";
  data += String(a.acceleration.z) + ",";

  data += String(g.gyro.x) + ",";
  data += String(g.gyro.y) + ",";
  data += String(g.gyro.z) + "\n";
  
  return data;
}

void connect(){
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    display.print(".");
    display.display();
    delay(500);
  }

  while (!client.connect("glovitoo-esp32", "acumalaka", "aezakmi123")) {
    Serial.print(".");
    display.print(".");
    display.display();
    delay(500);
  }

  
  client.subscribe("glovitoo/subtester");
  client.subscribe("glovitoo/speech");
}

void setup() {
  Serial.begin(115200);

  //OLED
  Serial.println("Beginning OLED");
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(("SSD1306 allocation failed"));
    for (;;)
      ;
  }
  Serial.println("OLED SUCCESS!");
  display.display();
  delay(500);
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setRotation(0);

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("OLED SUCCESS!");
  display.display();
  delay(500);
  ////////////////////////////////

  //MPU sensor
  Serial.println("Beginning MPU");
  if (!mpu.begin()) {
    Serial.println("Sensor init failed");
    display.display();
    while (1)
      yield();
  }
  delay(1000);
  ////////////////////////////////

  //WIFI&MQTT
  Serial.println("Begin WIFI and MQTT");
  WiFi.begin(SSID, PASS);

  client.begin("10.42.0.1", net);
  client.onMessage(messageReceived);

  connect();

}

void loop() {
  if((millis() - lastMillis) >10){
    client.loop();
    if (!client.connected()) {
      connect();
    }

    if((millis() - lastSensorSend) > 50){
      client.publish("glovitoo/sensors", allSensor());
      lastSensorSend=millis();
    }

    lastMillis = millis();
  }

}


void messageReceived(String &topic, String &payload){
  Serial.println("Payload masuk dari topik: " + topic + " dengan payload: " + payload);
}
