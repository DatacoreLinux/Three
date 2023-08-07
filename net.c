#include "n.h"
#include "glyphs.c"

//GRAPHICS
#define WINDOW_WIDTH 2560
#define WINDOW_HEIGHT 1440
#define WINDOW_X -10
#define WINDOW_Y -35


//SPECIFY PARAMETERS
#define H 0.0375
#define RATE 20.0
#define EPOCHS 1500
#define ARCH 1,3,6,9,14*14

//GLOBAL
size_t arch[] = {ARCH};
float global_rate = RATE;
bool paused = false;
float selected = 0;


#define gotoxy(x,y) printf("\033[%d;%dH", (y), (x))

#define ARCH_COUNT(arch) (sizeof(arch)/sizeof(arch[0]))

#define BUFFER_SIZE 150

struct buffer {
        float data;
        struct buffer *next;


};
typedef struct buffer Buffer;
Buffer *start = NULL;
Buffer *last = NULL;
Buffer *track = NULL;

void insert_buffer_element(float data)
{
        Buffer *new = malloc(sizeof(Buffer));
        new->data = data;

        if(start == NULL) {
                start = new;
                new->next = NULL;
                last = new;
        }
        else {
                last->next = new;
                new->next = NULL;
                last = new;
        }

        return;
}


void init_buffer(int num_elements, float set)
{
        for(int i = 0; i < num_elements; i++) {
                insert_buffer_element(set);
        }
        last->next = start;

        return;
}

void free_buffer()
{
        track = start;
        Buffer *temp = NULL;
        for(int i = 0; i < BUFFER_SIZE; i++) {
                temp = track;
                track = track->next;
                free(temp);
        }
        start = NULL;
        last = NULL;
        track = NULL;
        return;
}



#define TRAIN_DIM 14
float train_data[][TRAIN_DIM] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,1,0,0,0,1,1,1,1,1,0},
    {0,0,1,0,0,1,0,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,1,1,1,1,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,0,0,1,0,0,0},
    {0,1,0,0,0,0,1,0,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0},

};
#define TRAIN_COUNT (TRAIN_DIM * TRAIN_DIM)

float train_data2[][TRAIN_DIM] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,1,1,1,1,1,1,1,1,1,1,1,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,0,0,1,0,1,0},
    {0,1,0,0,0,0,0,0,0,1,0,0,1,0},
    {0,1,0,0,0,0,0,0,1,0,0,0,1,0},
    {0,1,0,1,0,0,0,1,0,0,0,0,1,0},
    {0,1,0,0,1,0,1,0,0,0,0,0,1,0},
    {0,1,0,0,0,1,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,1,1,1,1,1,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0},

};

float train_data3[][TRAIN_DIM] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,1,1,1,1,1,1,1,1,1,1,1,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,1,0,1,0,0,0,0,0,0,1,0,1,0},
    {0,1,0,0,1,0,0,0,0,1,0,0,1,0},
    {0,1,0,0,0,1,0,0,1,0,0,0,1,0},
    {0,1,0,0,0,0,1,1,0,0,0,0,1,0},
    {0,1,0,0,0,0,1,1,0,0,0,0,1,0},
    {0,1,0,0,0,1,0,0,1,0,0,0,1,0},
    {0,1,0,0,1,0,0,0,0,1,0,0,1,0},
    {0,1,0,1,0,0,0,0,0,0,1,0,1,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,1,1,1,1,1,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0},

};


typedef struct {

    size_t layers;
    Matrix x;
    Matrix *w;
    Matrix *b;
    Matrix *a;

} Network;

#define NETWORK_OUT_MATRIX n.a[n.layers - 1]

Network network_alloc(size_t *arch, size_t arch_size)
{
    Network n;

    n.layers = arch_size - 1;

    n.w = malloc(sizeof(*n.w) * n.layers);
    n.b = malloc(sizeof(*n.w) * n.layers);
    n.a = malloc(sizeof(*n.w) * n.layers);

    //ALLOCATE INPUT MATRIX
    n.x = matrix_alloc(arch[0], 1);

    //ALLOCATE weight, bias and activations for each layer
    for(size_t i = 0; i < n.layers; i++) {
        n.w[i] = matrix_alloc(arch[i+1], arch[i]);
        matrix_rand(n.w[i]);

        n.b[i] = matrix_alloc(arch[i+1], 1);
        matrix_rand(n.b[i]);

        n.a[i] = matrix_alloc(arch[i+1], 1);
        
    }

    return n;
}

void network_print(Network n)
{    
    printf("\n\nInputs\n");
    matrix_print(n.x);
    for(size_t i = 0; i < n.layers; i++) {
        printf("\nLAYER: %d", i);
        printf("\nW\n");
        matrix_print(n.w[i]);
        printf("\nB\n");
        matrix_print(n.b[i]);
        printf("\nA\n");
        matrix_print(n.a[i]);
    }

    return;
}

void forward(Network n)
{
    //Forward inputs to first layer activation
    //Layer 0
    matrix_dot(n.a[0], n.x, n.w[0]);
    matrix_sum(n.a[0], n.b[0]);
    matrix_sig(n.a[0]);

    //Forward Remaining network
    //Layer 1 -> Output
    for(size_t i = 1; i < n.layers; i++) {
        matrix_dot(n.a[i],n.a[i - 1],n.w[i]);
        matrix_sum(n.a[i],n.b[i]);
        matrix_sig(n.a[i]);
    }

    return;
}

float cost(Network n)
{
    float result = 0.0f;

    MATRIX_ENTRY(n.x, 0, 0) = 0;
    forward(n);

    for(size_t i = 0; i < TRAIN_DIM; i++) {
        for(size_t j = 0; j < TRAIN_DIM; j++) {

            float y = MATRIX_ENTRY(NETWORK_OUT_MATRIX, (i * TRAIN_DIM) + j, 0); 
            float out = train_data[i][j];

            float d = y - out;
            result += d * d;
        }
    }

    MATRIX_ENTRY(n.x, 0, 0) = 1;
    forward(n);

    for(size_t i = 0; i < TRAIN_DIM; i++) {
        for(size_t j = 0; j < TRAIN_DIM; j++) {

            float y = MATRIX_ENTRY(NETWORK_OUT_MATRIX, (i * TRAIN_DIM) + j, 0); 
            float out = train_data2[i][j];

            float d = y - out;
            result += d * d;
        }
    }

    MATRIX_ENTRY(n.x, 0, 0) = 2;
    forward(n);

    for(size_t i = 0; i < TRAIN_DIM; i++) {
        for(size_t j = 0; j < TRAIN_DIM; j++) {

            float y = MATRIX_ENTRY(NETWORK_OUT_MATRIX, (i * TRAIN_DIM) + j, 0); 
            float out = train_data3[i][j];

            float d = y - out;
            result += d * d;
        }
    }


    return result / (TRAIN_COUNT * 3);
}

void train(Network n, Network g)
{   
    float c = cost(n);
    float save;
    float h = H;
    float rate = global_rate;

    for(int k = 0; k < n.layers; k++) {
        //Calc Gradient, Weights
        for(size_t i = 0; i < n.w[k].rows; i++) {
            for(size_t j = 0; j < n.w[k].cols; j++) {
                save = MATRIX_ENTRY(n.w[k], i, j);
                MATRIX_ENTRY(n.w[k], i, j) += h;
                MATRIX_ENTRY(g.w[k],i,j) = ( cost(n) - c ) / h;
                MATRIX_ENTRY(n.w[k], i, j) = save;            
            }
        }

        //Calc Gradient, Biases
        for(size_t i = 0; i < n.b[k].rows; i++) {
            for(size_t j = 0; j < n.b[k].cols; j++) {
                save = MATRIX_ENTRY(n.b[k], i, j);
                MATRIX_ENTRY(n.b[k], i, j) += h;
                MATRIX_ENTRY(g.b[k],i,j) = ( cost(n) - c ) / h;
                MATRIX_ENTRY(n.b[k], i, j) = save;            
            }
        }
    }

    for(int k = 0; k < n.layers; k++) {

        //Apply Gradient, Weights
        for(size_t i = 0; i < n.w[k].rows; i++) {
            for(size_t j = 0; j < n.w[k].cols; j++) {
                MATRIX_ENTRY(n.w[k], i, j) -= MATRIX_ENTRY(g.w[k], i, j) * rate;
            }
        }

        //Apply Gradient, Biases
        for(size_t i = 0; i < n.b[k].rows; i++) {
            for(size_t j = 0; j < n.b[k].cols; j++) {
                MATRIX_ENTRY(n.b[k], i, j) -= MATRIX_ENTRY(g.b[k], i, j) * rate;
            }
        }
    }

    return;
}

void verify(Network n) {

    MATRIX_ENTRY(n.x, 0, 0) = 0;
    forward(n);

    for(size_t i = 0; i < TRAIN_DIM; i++) {
        for(size_t j = 0; j < TRAIN_DIM; j++) {

            int y = round(MATRIX_ENTRY(NETWORK_OUT_MATRIX, (i * TRAIN_DIM) + j, 0)); 

            if(y == 1) {
                printf("# ");
            }
            else {
                printf("  ");
            }
        }
        printf("\n");
    }


    MATRIX_ENTRY(n.x, 0, 0) = 1;
    forward(n);

    for(size_t i = 0; i < TRAIN_DIM; i++) {
        for(size_t j = 0; j < TRAIN_DIM; j++) {

            int y = round(MATRIX_ENTRY(NETWORK_OUT_MATRIX, (i * TRAIN_DIM) + j, 0)); 

            if(y == 1) {
                printf("# ");
            }
            else {
                printf("  ");
            }
        }
        printf("\n");
    }
}

void output(Network n, float select)
{
    MATRIX_ENTRY(n.x, 0, 0) = select;
    forward(n);

    for(size_t i = 0; i < TRAIN_DIM; i++) {
        for(size_t j = 0; j < TRAIN_DIM; j++) {

            int y = round(MATRIX_ENTRY(NETWORK_OUT_MATRIX, (i * TRAIN_DIM) + j, 0)); 

            if(y == 1) {
                printf("# ");
            }
            else {
                printf("  ");
            }
        }
        printf("\n");
    }

}

void SDL_RenderFillCircle(SDL_Renderer *renderer, int x, int y, int radius, SDL_Color color)
{
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    for (int w = 0; w < radius * 2; w++)
    {
        for (int h = 0; h < radius * 2; h++)
        {
            int dx = radius - w; // horizontal offset
            int dy = radius - h; // vertical offset
            if ((dx*dx + dy*dy) <= (radius * radius))
            {
                SDL_RenderDrawPoint(renderer, x + dx, y + dy);
            }
        }
    }
}

void render_network(SDL_Renderer *renderer, size_t *arch, int arch_count, int x, int y, int w, int h, Network n)
{

    if(paused == true) {
        MATRIX_ENTRY(n.x, 0, 0) = selected;
        forward(n);
    }

    //CALC LAYERS MID POINTS
    float layer_mid[arch_count];

    float layer_width = (float)w / (float)arch_count;

    for(size_t i = 0; i < arch_count; i++) {
        layer_mid[i] = (layer_width / 2) + (i * layer_width);
        //printf("\nLAYER MID: %d", layer_mid[i]);
    }

    float layer_step[arch_count];

    for(size_t i = 0; i < arch_count; i++) {
        layer_step[i] = ((float)h / (float)arch[i]);
    }

    SDL_Color c = {0x80,0xe0,0xff,255};

    for(size_t i = 0; i < arch_count; i++) {

        for(int j = 0; j < arch[i]; j++) {
        
            float x1 = layer_mid[i]; 
            float y1 =  0 + (float)layer_step[i] * (float)j + ((float)layer_step[i] / 2.f); 
            float size = (100.f / (float)arch[i]) + (float)4;

            if(i < arch_count -1) {
                for(int k = 0; k < arch[i+1]; k++) {
                    float x2 = layer_mid[i+1];
                    float y2 =  0 + (float)layer_step[i+1] * (float)k + ((float)layer_step[i+1] / 2.f); 
                    //SDL_SetRenderDrawColor(renderer, 0xff, 0xff, 0x00, 255);
                    //SDL_SetRenderDrawColor(renderer, 20 * j, 20 * k, 20 * i, 255);
                    //SDL_SetRenderDrawColor(renderer, (255.f / (float)arch[i+1]) * (float)k, (255.f / (float)arch[i+1]) * j, (255.f / (float)arch[i+1]) * i, 255);
                    SDL_SetRenderDrawColor(renderer, 0x55, 0x55, 0x55, 255);

                    


                    float bright = 0.f;
                    if(i != 0) {
                        bright = MATRIX_ENTRY(n.w[i-1], k, 0) ;
                    }


                    SDL_SetRenderDrawColor(renderer, 0x88, 0x88, 0x88, (155 * bright) + 100);

                    SDL_RenderDrawLine(renderer, x1 + x, y1 + y, x2 + x, y2 + y );
                        
                }
            }

            float scale = 1;
            if(i != 0) {
                scale = MATRIX_ENTRY(n.a[i-1], j, 0);
            }


            float glow = 1;
            if(i != 0) {
                glow = MATRIX_ENTRY(n.b[i-1], j, 0) + 1;
            }


            for(int i = 20; i > 0; i =- 2) {
                SDL_Color c5 = {0x80,0xe0,0xff, i};
                SDL_RenderFillCircle(renderer, x1 + x ,y1 + y, (size * scale) + i * glow, c5);

            }

            SDL_RenderFillCircle(renderer, x1 + x ,y1 + y, size * scale, c);
            //SDL_RenderFillCircle(renderer, x1 + x ,y1 + y, size, c);
        }

    }

    return;
}

void render_image(SDL_Renderer *renderer, int x, int y, int w, int h, Network n, float input, SDL_Color c)
{
    SDL_Rect rect = {
        x,
        y,
        h,
        h
    };



    SDL_SetRenderDrawColor(renderer,0x55,0x55,0x55,255);

    SDL_RenderDrawRect(renderer, &rect);

    float cellSize = (float)w / 14.f;
    float offset = 15;

    MATRIX_ENTRY(n.x, 0, 0) = input;
    forward(n);
     
    for(int i = 0; i < 14; i++) {
        for(int j = 0; j < 14; j++) {

            //float out = MATRIX_ENTRY(NETWORK_OUT_MATRIX, (j*TRAIN_DIM)+i, 0);
            int out = round(MATRIX_ENTRY(NETWORK_OUT_MATRIX, (j*TRAIN_DIM)+i, 0));

            float size = 10.f * out;
            c.a = 255.f * out;
            SDL_RenderFillCircle(renderer, x + (i * cellSize) + offset , y + (j * cellSize) + offset, size, c);
        }
    }
}

void render_images(SDL_Renderer *renderer, int x, int y, int w, int h, Network n)
{
    SDL_Color c; c.r = 0xff; c.g = 0xff; c.b = 0x00; c.a = 255;
    render_image(renderer,x,y,h/3,h/3,n, 0,c);

    c; c.r = 0x80; c.g = 0xe0; c.b = 0xff; c.a = 255;
    render_image(renderer,x,y + (h/3) + 20,h/3,h/3,n,1,c);

    c; c.r = 0xff; c.g = 0x00; c.b = 0xff; c.a = 255;
    render_image(renderer,x,y + (h/3) + (h/3) + 40,h/3,h/3,n,2,c);

    return;
}

void render_images_select(SDL_Renderer *renderer, int x, int y, int w, int h, Network n)
{
    SDL_Color c; c.r = 0xff; c.g = 0xff; c.b = 0x00; c.a = 255;
    render_image(renderer,x,y,h/3,h/3,n, selected,c);

    c; c.r = 0x80; c.g = 0xe0; c.b = 0xff; c.a = 255;
    render_image(renderer,x,y + (h/3) + 20,h/3,h/3,n,selected,c);

    c; c.r = 0xff; c.g = 0x00; c.b = 0xff; c.a = 255;
    render_image(renderer,x,y + (h/3) + (h/3) + 40,h/3,h/3,n,selected,c);

    return;
}

void render_cost(SDL_Renderer *renderer, int x, int y, int w, int h)
{
    SDL_Rect rect = {
        x,
        y,
        h,
        h
    };



    SDL_SetRenderDrawColor(renderer,0x55,0x55,0x55,255);
    SDL_RenderDrawRect(renderer, &rect);

    rect.w = 5;
    rect.h = 5;

    SDL_SetRenderDrawColor(renderer,0xff,0xff,0x00,255);



    //float xstep = ((float)w / (float)cost_count);
    //
    float xstep = ((float)w / BUFFER_SIZE);



    Buffer *temp = track;
    for(int i = 0; i < BUFFER_SIZE; i++) {

        rect.x = x + (i * xstep);
        //rect.y =   ((float)y + ((float)y/2)) - (cost_graph[i] * h);
        rect.y =   ((float)y * 2.2) - (temp->data * h);

        SDL_SetRenderDrawColor(renderer, 0 + (temp->data * 250) , 255 - (temp->data * 250)  ,0x00,255);

        temp = temp->next;


        SDL_RenderFillRect(renderer, &rect);

    }

    return;
}

int epoch_count = 0;

void render_cost_val(SDL_Renderer *renderer, int x, int y, int w, int h, Network n)
{
    epoch_count++;

    SDL_Rect rect = {
        x,
        y,
        w,
        h
    };



    SDL_SetRenderDrawColor(renderer,0x55,0x55,0x55,255);
    SDL_RenderDrawRect(renderer, &rect);

    char buffer[9];
    int ret = snprintf(buffer, sizeof(buffer), "%f", cost(n));



    SDL_Color c = {0x80,0xe0,0xff,255};
    SDL_Color c2 = {0x11,0x11,0x11,255};

    SDL_Color c3 = {0xff,0x00,0xff,50};


    
    float glyph_width = w / 8;
    float glyph_height = h / 5;

    float xstep = glyph_width / 9;
    float ystep = glyph_height / 10;

    float padding = 5;

    float cost_padding = 230;

    for(int k = 0; k < 8; k++) {

        int g = buffer[k];

        for(int i = 0; i < 10; i++) {
            for(int j = 0; j < 9; j++) {

                if(glyphs[g][i][j] == 1) {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 4, c3);
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 2, c);


                }
                else {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 1, c2);

                }

            }
        }
    }


    char buffer2[5] = "COST";
    //int ret2 = snprintf(buffer, sizeof(buffer), "%f", "");


    glyph_width = w / 4;
    glyph_height = h / 2.5;

    xstep = glyph_width / 9;
    ystep = glyph_height / 10;


    for(int k = 0; k < 5; k++) {

        int g = buffer2[k];

        for(int i = 0; i < 10; i++) {
            for(int j = 0; j < 9; j++) {

                if(glyphs[g][i][j] == 1) {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding, 4, c3);
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding, 3, c);


                }
                else {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 1, c2);

                }

            }
        }
    }




    return;
}

void render_rate(SDL_Renderer *renderer, int x, int y, int w, int h)
{
    SDL_Rect rect = {
        x,
        y,
        w,
        h
    };



    SDL_SetRenderDrawColor(renderer,0x55,0x55,0x55,255);
    SDL_RenderDrawRect(renderer, &rect);


    char buffer[9];
    int ret = snprintf(buffer, sizeof(buffer), "%f", global_rate);



    SDL_Color c = {0x80,0xe0,0xff,255};
    SDL_Color c2 = {0x11,0x11,0x11,255};

    SDL_Color c3 = {0xff,0x00,0xff,50};


    
    float glyph_width = w / 8;
    float glyph_height = h / 5;

    float xstep = glyph_width / 9;
    float ystep = glyph_height / 10;

    float padding = 5;

    float cost_padding = 230;

    for(int k = 0; k < 8; k++) {

        int g = buffer[k];

        for(int i = 0; i < 10; i++) {
            for(int j = 0; j < 9; j++) {

                if(glyphs[g][i][j] == 1) {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 4, c3);
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 2, c);


                }
                else {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 1, c2);

                }

            }
        }

    }


    char buffer2[5] = "RATE";
    //int ret2 = snprintf(buffer, sizeof(buffer), "%f", "");


    glyph_width = w / 4;
    glyph_height = h / 2.5;

    xstep = glyph_width / 9;
    ystep = glyph_height / 10;


    for(int k = 0; k < 5; k++) {

        int g = buffer2[k];

        for(int i = 0; i < 10; i++) {
            for(int j = 0; j < 9; j++) {

                if(glyphs[g][i][j] == 1) {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding, 4, c3);
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding, 3, c);


                }
                else {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding, y + (ystep * i) + padding + cost_padding, 1, c2);

                }

            }
        }
    }



    return;
}

void render_paused(SDL_Renderer *renderer, int x, int y, int w, int h)
{
    SDL_Rect rect = {
        x,
        y,
        w,
        h
    };



    SDL_SetRenderDrawColor(renderer,0x55,0x55,0x55,255);
    SDL_RenderDrawRect(renderer, &rect);


    char buffer[7] = "PAUSED";




    SDL_Color c = {0xFF,0x00,0x00,255};
    SDL_Color c2 = {0x11,0x11,0x11,255};

    SDL_Color c3 = {0xff,0x00,0xff,50};


    
    float glyph_width = w / 8;
    float glyph_height = h / 1;

    float xstep = glyph_width / 9;
    float ystep = glyph_height / 10;

    float padding = 5;
    float indent = 70;


    for(int k = 0; k < 8; k++) {

        int g = buffer[k];

        for(int i = 0; i < 10; i++) {
            for(int j = 0; j < 9; j++) {

                if(glyphs[g][i][j] == 1) {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding + indent, y + (ystep * i) + padding, 4, c3);
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding + indent, y + (ystep * i) + padding, 2, c);


                }
                else {
                    SDL_RenderFillCircle(renderer, x + (xstep * j) + (k * glyph_width) + padding + indent, y + (ystep * i) + padding, 1, c2);

                }

            }
        }

    }


    return;
}

void render_screen(SDL_Renderer *renderer, Network n)
{

    SDL_SetRenderDrawColor(renderer, 0x11, 0x11, 0x11, 255);
    //SDL_RenderClear(renderer);
    //SDL_RenderPresent(renderer);


    int Nx = 600;
    int Ny = 200;
    int Nw = 1300;
    int Nh = 1000;
    

    render_network(renderer, arch, ARCH_COUNT(arch), Nx, Ny, Nw, Nh,n);


    int Ix = 1900;
    int Iy = 50;
    int Iw = 500;
    int Ih = 1300;
    
    if(paused == false) {
        render_images(renderer, Ix, Iy, Iw, Ih, n);
    }
    else {
        render_images_select(renderer, Ix, Iy, Iw, Ih, n);

    }

    int Cx = 50;
    int Cy = (WINDOW_HEIGHT - 550.f) / 2.f;
    int Cw = 550;
    int Ch = 550;

    render_cost(renderer, Cx, Cy, Cw, Ch);

    int CVx = 50;
    int CVy = 50;
    int CVw = 550;
    int CVh = 300;

    render_cost_val(renderer, CVx, CVy, CVw, CVh, n);


    int Rx = 50;
    int Ry = 1100;
    int Rw = 550;
    int Rh = 300;

    render_rate(renderer, Rx, Ry, Rw, Rh);


    int Px = 950;
    int Py = 1330;
    int Pw = 550;
    int Ph = 75;

    if(paused == true) {
        render_paused(renderer, Px, Py, Pw, Ph);
    }

}





int main() 
{
    srand(time(0));



    




    SDL_Window *window;
    SDL_Renderer *renderer;

    if(SDL_INIT_VIDEO < 0) {
        fprintf(stderr,"ERROR: SDL_INIT_VIDEO");
    }
    
    window = SDL_CreateWindow(
        "",
        WINDOW_X,
        WINDOW_Y,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_BORDERLESS
    );  


    if(!window) {
        fprintf(stderr,"ERROR: !window");
    }




    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);


    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);


    //Declare and allocate network and gradient mirror
    Network n = network_alloc(arch, ARCH_COUNT(arch));
    Network g = network_alloc(arch, ARCH_COUNT(arch));

    //COST BUFFER
    init_buffer(BUFFER_SIZE, cost(n));



    SDL_Event event;
    bool quit = false;



    track = start;
    while(!quit) {

        if(paused == false) {
            cost(n);
            train(n,g);
        }

        while(SDL_PollEvent(&event)) {




            switch(event.type) {
                case SDL_QUIT:
                    quit = true;
                    break;
                case SDL_KEYUP:
                    break;
                case SDL_KEYDOWN:
                    switch(event.key.keysym.sym) {
                        case SDLK_ESCAPE:
                            quit = true;
                            break;
                        case SDLK_LEFT:
                            if(global_rate > 1) {
                                global_rate--;
                            }
                            else {
                                global_rate = global_rate / 10;
                            }
                            break;
                        case SDLK_RIGHT:
                            if(global_rate > 0.999999) {
                                global_rate++;
                            }
                            else {
                                global_rate = global_rate * 10;
                            }
                            break;
                        case SDLK_SPACE:
                            if(paused == false) {
                                paused = true;
                            }
                            else {
                                paused = false;

                            }
                            break;
                        case SDLK_0:
                            if(paused == true) {
                                selected = 0;
                            }
                            break;
                        case SDLK_1:
                            if(paused == true) {
                                selected = 1;
                            }
                            break;
                        case SDLK_2:
                            if(paused == true) {
                                selected = 2;
                            }
                            break;

                    }



                    break;
            }

        }



        
        SDL_SetRenderDrawColor(renderer, 0x11, 0x11, 0x11, 255);
        SDL_RenderClear(renderer);
        track->data = cost(n);
        track = track->next;
        //gotoxy(0,0);
        render_screen(renderer, n);

        //cost(n);
        //verify(n);


        SDL_RenderPresent(renderer);
    }





    //Specify Architecture for Network



/*

    system("clear");
    //Training Loop
    for(size_t i = 0; i < EPOCHS; i++) {
        gotoxy(0,0);
        train(n,g);

        printf("\nCOST: %f EPOCHS %d : %d\n", cost(n) , i , EPOCHS); 
        verify(n);
    }

    //Test Trained Network
    system("clear");
    float select;
    while(1) {
        gotoxy(0,0);
        printf("Enter a number (0 or 1) :");
        scanf("%f", &select);
        output(n, select);
    }
*/
    return 0;
}
