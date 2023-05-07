#include <Arduino.h>

const int board_id = 1; // Set a unique ID for each board (1 to 5)
const int NUM_FEATURES = 6;
float weights[NUM_FEATURES] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

const int ledPin = 13; // Built-in LED pin number (usually 13)

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

void sendWeights() {
  Serial.print("weights,");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.print(weights[i], 2);
    if (i < NUM_FEATURES - 1) {
      Serial.print(",");
    }
  }
  Serial.println();
}

void setup() {
  pinMode(ledPin, OUTPUT); // Set the LED pin as an output
  Serial.begin(9600);
  unsigned long start_time = millis();
  while (!Serial && (millis() - start_time < 5000)) {
    ; // wait for serial port to connect. Needed for native USB, timeout after 5 seconds
  }
}

void loop() {
  Serial.println("Loop is running"); // Debug message
  if (Serial.available()) {
    int received_id = Serial.parseInt();
    String command = Serial.readStringUntil(',');

    Serial.print("Received ID: "); // Debug message
    Serial.println(received_id); // Debug message
    Serial.print("Received command: "); // Debug message
    Serial.println(command); // Debug message

    if (received_id == board_id) {
      if (command == "train") {
        Serial.println("Received train command");

        Serial.println("Turning LED on");
        digitalWrite(ledPin, HIGH); // Turn on the LED
        Serial.println("LED should be on now");

        int epochs = Serial.parseInt();
        float learning_rate = Serial.parseFloat();
        int num_samples = Serial.parseInt();
        int batch_size = Serial.parseInt();

        for (int epoch = 0; epoch < epochs; epoch++) {
          float gradients[NUM_FEATURES] = {0, 0, 0, 0, 0, 0};

          // Process samples in batches
          for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            for (int i = 0; i < batch_size; i++) {
              float features[NUM_FEATURES];
              int label;

              String datapoint = Serial.readStringUntil('\n');
              String parts[NUM_FEATURES + 1];
              int partIdx = 0;

              int idx = datapoint.indexOf(',');
              while (idx != -1) {
                parts[partIdx++] = datapoint.substring(0, idx);
                datapoint = datapoint.substring(idx + 1);
                idx = datapoint.indexOf(',');
              }
              parts[partIdx] = datapoint;

              label = parts[0].toInt();
              for (int j = 0; j < NUM_FEATURES; j++) {
                features[j] = parts[j + 1].toFloat();
              }

              float z = 0;
              for (int j = 0; j < NUM_FEATURES; j++) {
                z += weights[j] * features[j];
              }

              float prediction = sigmoid(z);
              float error = prediction - label;

              for (int j = 0; j < NUM_FEATURES; j++) {
                gradients[j] += error * features[j];
              }
            }
          }

          // Update
          for (int i = 0; i < NUM_FEATURES; i++) {
            weights[i] -= learning_rate * gradients[i] / num_samples;
          }
        }

        digitalWrite(ledPin, LOW); // Turn off the LED after training is complete
        sendWeights(); // Send the updated weights back to Python
      }
    }
  }
}
