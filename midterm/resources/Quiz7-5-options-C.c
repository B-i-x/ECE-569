for (int i = 1; i < ARRAY_SIZE; i++) {
    acc = acc + elements[i + 1];
    out[i + 1] = acc;
}