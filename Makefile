CLFAGS := -O2
CINCLUDE := -Iinclude

all: sort generate_network
SRC := $(addprefix src/, bitonic.c helper.c main.c)

generate_network: src/generate_network.c
	$(CC) $(CLFAGS) $^ -o $@

sort: $(SRC)
	$(CC) $(CLFAGS) $(CINCLUDE) $(SRC) -o sort

sort_network.o: ./sort_network/sort_newtork.c
	$(CC) $(CLFAGS) $^ -c -o $@

test_network: src/sortnet_test.c sort_network.o src/helper.c
	$(CC) $(CFLAGS) $(CINCLUDE) $^ -o $@

debug: $(SRC)
	$(CC) $(CLFAGS) -g $(CINCLUDE) $(SRC) -o sort

.PHONY: clean
clean:
	rm sort generate_network test_network sort_network.o ./sort_network/sort_newtork.c