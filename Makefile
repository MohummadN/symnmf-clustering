CC      = gcc
CFLAGS  = -ansi -Wall -Wextra -Werror -pedantic-errors -O2
LDLIBS  = -lm
OBJS    = main.o symnmf.o io.o

.PHONY: all ext test clean

all: symnmf

symnmf: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

%.o: %.c symnmf.h
	$(CC) $(CFLAGS) -c $<

ext:
	python3 setup.py build_ext --inplace

test: all ext
	python3 -m pytest -q tests

clean:
	rm -rf *.o symnmf build *.so __pycache__ tests/__pycache__ .pytest_cache
