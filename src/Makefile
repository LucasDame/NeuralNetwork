# Compilateur
CC = gcc

# Flags (Optimisation + Warnings)
CFLAGS = -Wall -Wextra -O3 -g

# Libs
LIBS = -lm

# Nom de l'exécutable
TARGET = neural_net

# Fichier principal
SRC = main.c

# Dépendances (Les fichiers que main.c inclut)
# Si un de ces fichiers change, on recompile !
DEPS = network.c layer.c matrix.c

all: $(TARGET)

# La règle magique : TARGET dépend de SRC ET de DEPS
$(TARGET): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)