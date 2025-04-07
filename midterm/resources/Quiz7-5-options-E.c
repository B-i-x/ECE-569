for (int i = 1; i < ARRAY_SIZE; i++) {
    out[i] = acc;
    acc = acc + elements[i];
}