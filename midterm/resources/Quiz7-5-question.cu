int main(int argc, char **argv) {
    const int ARRAY_SIZE = 8;
    int acc = 0;
    int out[ARRAY_SIZE];
    int elements[] = {0, 3, 4, 11, 11, 15, 16, 22};

    for (int i = 0; i < ARRAY_SIZE; i++) {
        acc = acc + elements[i];
        out[i] = acc;
    }
}