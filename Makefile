CC = gcc
CFLAGS = 
LIBS = -lSDL2 -lm
INCLUDE = 
SRCS = net.c
OBJS = $(SRCS:.c=.o)
MAIN = net

$(MAIN) : $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDE) -o $(MAIN) $(OBJS) $(LIBS)

clean:
	$(RM) *.o *~ $(MAIN)
