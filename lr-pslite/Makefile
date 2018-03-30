INC_DIR=./include
BIN_DIR=./bin
SRC_DIR=./src
OBJ_DIR=./obj

SRC=${wildcard ${SRC_DIR}/*.cc}
OBJ=${patsubst %.cc, $(OBJ_DIR)/%.o, ${notdir ${SRC}}}

TARGET=lr
BIN_TARGET=${BIN_DIR}/${TARGET}

CC=g++
CFLAGS=-std=c++11 -g -Wall -I${INC_DIR}

all:${BIN_TARGET}

${BIN_TARGET}: ${OBJ}
	${CC} ${OBJ} -o $@

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cc
	${CC} ${CFLAGS} -c $< -o $@

test:
	echo $(SRC)
	echo $(OBJ)
	echo $(CFLAGS)
	echo $(CC)

