# Compilateur
CC = gcc

# Flags (Optimisation + Warnings)
CFLAGS = -Wall -Wextra -O3 -g

# Libs
LIBS = -lm

# Nom de l'exécutable
TARGET = src/neural_net

# Fichier principal
SRC = src/main.c

# Dépendances (Les fichiers que main.c inclut)
# Si un de ces fichiers change, on recompile !
DEPS = src/network.c src/layer.c src/matrix.c

all: $(TARGET)

# La règle magique : TARGET dépend de SRC ET de DEPS
$(TARGET): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)