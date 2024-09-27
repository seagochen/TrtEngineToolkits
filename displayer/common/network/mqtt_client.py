import paho.mqtt.client as mqtt
from typing import Optional, Callable
from common.utils.logger import Logger

logger = Logger()

class MQTTClient:
    def __init__(self, address: str, port: int, client_id: str, username: Optional[str] = None, password: Optional[str] = None):
        self.address = address
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311, transport="tcp")
        self.is_connected = False
        self.message_callback = None

        # 设置用户名和密码
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        # 设置回调函数
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

    def set_message_callback(self, callback: Callable[[str, bytes, int], None]):
        """设置接收消息的回调函数"""
        self.message_callback = callback

    def connect(self) -> bool:
        """连接到MQTT broker"""
        try:
            self.client.connect(self.address, self.port, 60)
            # 启动事件循环线程
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error("MQTTClient", f"Failed to connect to {self.address}:{self.port} - {str(e)}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.is_connected:
            self.client.disconnect()
            self.client.loop_stop()

    def publish(self, topic: str, payload: bytes) -> bool:
        """发布消息"""
        try:
            self.client.publish(topic, payload)
            return True
        except Exception as e:
            logger.error("MQTTClient", f"Failed to publish message to {topic} - {str(e)}")
            return False

    def subscribe(self, topic: str) -> bool:
        """订阅主题"""
        try:
            self.client.subscribe(topic)
            return True
        except Exception as e:
            logger.error("MQTTClient", f"Failed to subscribe to {topic} - {str(e)}")
            return False

    # 回调函数
    def on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            logger.info("MQTTClient", f"Connected to the MQTT broker at {self.address}:{self.port}")
            self.is_connected = True
        else:
            logger.error("MQTTClient", f"Failed to connect: {mqtt.error_string(rc)}")

    def on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        if rc == 0:
            logger.info("MQTTClient", f"Disconnected from the MQTT broker gracefully.")
        else:
            logger.error("MQTTClient", f"Unexpected disconnection: {mqtt.error_string(rc)}")
        self.is_connected = False

    def on_message(self, client, userdata, msg):
        """消息回调"""
        if self.message_callback:
            self.message_callback(msg.topic, msg.payload, len(msg.payload))

# Example usage
if __name__ == "__main__":

    def message_handler(topic, payload, payloadlen):
        print(f"Received message on topic '{topic}': {payload.decode('utf-8')}")

    # Initialize the MQTT client
    mqtt_client = MQTTClient(address="localhost", port=1883, client_id="test_client", username="", password="")

    # Set message callback
    mqtt_client.set_message_callback(message_handler)

    # Connect to the broker
    if mqtt_client.connect():
        # Subscribe to a topic
        mqtt_client.subscribe("test/topic")

        # Publish a message
        mqtt_client.publish("test/topic", b"Hello, MQTT!")

        # Wait for messages
        try:
            while True:
                pass
        except KeyboardInterrupt:
            # Disconnect when exiting
            mqtt_client.disconnect()
