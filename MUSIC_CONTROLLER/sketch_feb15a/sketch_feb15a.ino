/*
  LED Music Sync - Arduino
  Recebe PWM via Serial e aplica 3 modos:
  - mode 0: fade forte (suave)
  - mode 1: snap imediato (sem fade)
  - mode 2: fade rápido de transição

  Protocolo serial (oficial, 6 canais):
  P,mode,L1,L2,L3,L4,L5,L6\n
  Compatibilidade:
  P,L1,L2,L3,L4,L5,L6\n (assume mode 0)
  P,mode,L1,L2,L3,L4\n   (legacy 4 canais)
  P,L1,L2,L3,L4\n        (legacy 4 canais, mode 0)
*/

const uint8_t NUM_LEDS = 6;
const uint8_t LED_PINS[NUM_LEDS] = {3, 5, 6, 9, 10, 11};

int currentPwm[NUM_LEDS] = {0, 0, 0, 0, 0, 0};
int targetPwm[NUM_LEDS] = {0, 0, 0, 0, 0, 0};
uint8_t outputMode = 0;

unsigned long lastUpdateMs = 0;
const uint16_t UPDATE_MS = 12;
const uint8_t FADE_STEP_SOFT = 4;  // mode 0
const uint8_t FADE_STEP_FAST = 9;  // mode 2

void clearTargets() {
  for (uint8_t i = 0; i < NUM_LEDS; i++) {
    targetPwm[i] = 0;
  }
}

void applyStartupEffect() {
  // Varredura inicial esquerda->direita->esquerda
  for (int pass = 0; pass < 2; pass++) {
    for (uint8_t i = 0; i < NUM_LEDS; i++) {
      for (uint8_t led = 0; led < NUM_LEDS; led++) {
        analogWrite(LED_PINS[led], led == i ? 180 : 0);
      }
      delay(70);
    }
    for (int i = NUM_LEDS - 2; i >= 1; i--) {
      for (uint8_t led = 0; led < NUM_LEDS; led++) {
        analogWrite(LED_PINS[led], led == i ? 180 : 0);
      }
      delay(70);
    }
  }

  for (uint8_t i = 0; i < NUM_LEDS; i++) {
    analogWrite(LED_PINS[i], 0);
  }
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_LEDS; i++) {
    pinMode(LED_PINS[i], OUTPUT);
    analogWrite(LED_PINS[i], 0);
  }

  applyStartupEffect();
  Serial.println("Arduino Ready (PWM mode 0/1/2, 6 LEDs)");
}

bool parseFrame(char* line) {
  // Espera oficial: P,mode,v1,v2,v3,v4,v5,v6
  // Compatível com versões antigas de 4 canais
  if (line[0] != 'P' || line[1] != ',') {
    return false;
  }

  const uint8_t MAX_VALUES = NUM_LEDS + 1;  // mode + 6
  int values[MAX_VALUES] = {0, 0, 0, 0, 0, 0, 0};

  char* token = strtok(line + 2, ",");
  uint8_t count = 0;

  while (token != NULL && count < MAX_VALUES) {
    values[count++] = atoi(token);
    token = strtok(NULL, ",");
  }

  if (count == (NUM_LEDS + 1)) {
    // P,mode,6valores
    outputMode = constrain(values[0], 0, 2);
    for (uint8_t i = 0; i < NUM_LEDS; i++) {
      targetPwm[i] = constrain(values[i + 1], 0, 255);
    }
    return true;
  }

  if (count == NUM_LEDS) {
    // P,6valores
    outputMode = 0;
    for (uint8_t i = 0; i < NUM_LEDS; i++) {
      targetPwm[i] = constrain(values[i], 0, 255);
    }
    return true;
  }

  if (count == 5) {
    // Legacy: P,mode,4valores
    outputMode = constrain(values[0], 0, 2);
    clearTargets();
    for (uint8_t i = 0; i < 4; i++) {
      targetPwm[i] = constrain(values[i + 1], 0, 255);
    }
    return true;
  }

  if (count == 4) {
    // Legacy: P,4valores
    outputMode = 0;
    clearTargets();
    for (uint8_t i = 0; i < 4; i++) {
      targetPwm[i] = constrain(values[i], 0, 255);
    }
    return true;
  }

  return false;
}

void applyRamp(uint8_t step) {
  for (uint8_t i = 0; i < NUM_LEDS; i++) {
    if (currentPwm[i] < targetPwm[i]) {
      currentPwm[i] = min(currentPwm[i] + step, targetPwm[i]);
    } else if (currentPwm[i] > targetPwm[i]) {
      currentPwm[i] = max(currentPwm[i] - step, targetPwm[i]);
    }

    analogWrite(LED_PINS[i], currentPwm[i]);
  }
}

void updateOutput() {
  unsigned long now = millis();
  if (now - lastUpdateMs < UPDATE_MS) {
    return;
  }
  lastUpdateMs = now;

  if (outputMode == 1) {
    // Snap imediato
    for (uint8_t i = 0; i < NUM_LEDS; i++) {
      currentPwm[i] = targetPwm[i];
      analogWrite(LED_PINS[i], currentPwm[i]);
    }
    return;
  }

  if (outputMode == 2) {
    applyRamp(FADE_STEP_FAST);
    return;
  }

  applyRamp(FADE_STEP_SOFT);
}

void loop() {
  static char buffer[96];
  static uint8_t pos = 0;
  static bool droppingLine = false;

  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (!droppingLine && pos > 0) {
        buffer[pos] = '\0';
        parseFrame(buffer);  // frame inválido é ignorado sem mudar estado
      }
      pos = 0;
      droppingLine = false;
      continue;
    }

    if (droppingLine) {
      continue;
    }

    if (pos < sizeof(buffer) - 1) {
      buffer[pos++] = c;
    } else {
      // Overflow: descarta o restante da linha atual
      pos = 0;
      droppingLine = true;
    }
  }

  updateOutput();
}
