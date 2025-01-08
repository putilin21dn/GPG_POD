#include <iostream>
#include <algorithm>
#include <cmath>

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

struct vec3 {
    double x;
    double y;
    double z;

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(double x, double y, double z) : x(x), y(y), z(z) {}
};

struct light_source_t {
    vec3 position;
    uchar4 color;
    double intensity;
};

__host__ __device__ vec3 operator+(vec3 a, vec3 b) {
    return vec3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__host__ __device__ vec3 operator-(vec3 a, vec3 b) {
    return vec3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__host__ __device__ vec3 operator-(vec3 a) {
    return vec3(-a.x, -a.y, -a.z);
}

__host__ __device__ vec3 operator*(vec3 a, double b) {
    return vec3(
        a.x * b,
        a.y * b,
        a.z * b
    );
}

__host__ __device__ vec3 operator/(const vec3& v, double k) {
    return {v.x / k, v.y / k, v.z / k};
}

__host__ __device__ vec3 operator*(vec3 a, vec3 b){

  return vec3(
      a.x * b.x,
      a.y * b.y,
      a.z * b.z
  );
}

__host__ __device__ double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 cross(vec3 a, vec3 b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__host__ __device__ double length(const vec3& v) {
    return sqrt(dot(v, v));
}


__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return vec3(
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z
    );
}

__host__ __device__ vec3 norm(const vec3& v) {
    double l = length(v);
    if (l < 1e-15) return vec3(0,0,0);
    return v / l;
}

struct texture_t {
    uchar4* data;       // Данные текстуры на хосте
    uchar4* dev_data;   // Данные текстуры на устройстве (GPU)
    int width;          // Ширина текстуры
    int height;         // Высота текстуры
};

struct polygon {
    vec3 a;           // Вершина A
    vec3 b;           // Вершина B
    vec3 c;           // Вершина C
    uchar4 color;     // Цвет полигона
    double transparency; // Прозрачность полигона
    double reflection;   // Коэффициент отражения
    vec3 normal;      // Нормаль к полигону
    bool has_texture;       // Флаг наличия текстуры
    texture_t texture;

    __host__ __device__ polygon() {}

    __host__ __device__ polygon(vec3 a, vec3 b, vec3 c, uchar4 color)
        : a(a), b(b), c(c), color(color), transparency(0.5), reflection(0), has_texture(false) {
        normal = norm(cross(b - a, c - a));

        vec3 center_polygon = (a + b + c) / 3.0;

        vec3 to_vertex = norm(a - center_polygon);

        if (dot(normal, to_vertex) < 0.0) {
            normal = -normal;
        }
    }
};

__host__ __device__ uchar4 sample_texture(const texture_t& texture, double u, double v) {
    u = fmin(fmax(u, 0.0), 1.0);
    v = fmin(fmax(v, 0.0), 1.0);

    int x = (int)(u * (texture.width - 1));
    int y = (int)(v * (texture.height - 1));

    return texture.dev_data[y * texture.width + x];
}

// Функция для пересечения луча 
__host__ __device__ bool ray_intersects_polygon(
    vec3 pos, vec3 dir, polygon poly, vec3* intersection_point, double* t_out
) {
    const double EPSILON = 1e-5;
    vec3 edge1 = poly.b - poly.a;
    vec3 edge2 = poly.c - poly.a;
    vec3 h = cross(dir, edge2);
    double a = dot(edge1, h);

    if (fabs(a) < EPSILON) {
        return false; 
    }

    double f = 1.0 / a;
    vec3 s = pos - poly.a;
    double u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return false;
    }

    vec3 q = cross(s, edge1);
    double v = f * dot(dir, q);

    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    double t = f * dot(edge2, q);

    if (t > -1e-3) {  
        *intersection_point = pos + dir * t;
        *t_out = t;
        return true;
    }

    return false;
}


// Функция для проверки, находится ли точка в тени
__host__ __device__ bool is_in_shadow(
    vec3 point, vec3 light_dir, vec3 light_pos, polygon* polygons, int polygons_cnt
) {
    vec3 shadow_ray_origin = point + light_dir * 1e-2; 
    double max_dist = length(light_pos - point); 
    double light_attenuation = 1.0; 

    for (int i = 0; i < polygons_cnt; i++) {
        vec3 temp_intersection;
        double t = INFINITY;

        if (ray_intersects_polygon(shadow_ray_origin, light_dir, polygons[i], &temp_intersection, &t)) {
            if (t > -1e-3 && t < max_dist + 1e-3) {
                double transparency = polygons[i].transparency;
                if (transparency > 0.0) {
                    light_attenuation *= (1.0 - transparency);
                    if (light_attenuation < 0.1) {
                        return true;
                    }
                    shadow_ray_origin = temp_intersection + light_dir * 1e-2;
                    continue; 
                }
                return true; 
            }
        }
    }

    return false; 
}


__host__ __device__ vec3 reflect(const vec3& dir, const vec3& normal) {
    return dir - normal * 2.0 * dot(dir, normal);
}

// Основная функция трассировки лучей
__host__ __device__ uchar4 ray(
    vec3 pos,                // Позиция начала луча
    vec3 dir,                // Направление луча
    light_source_t* lights,  // Массив источников света
    int lights_cnt,          // Количество источников света
    polygon* polygons,       // Массив полигонов
    int polygons_cnt,        // Количество полигонов
    vec3 camera_position,    // Позиция камеры
    int depth                // Глубина рекурсии
) {
    vec3 color = vec3(0, 0, 0);
    vec3 attenuation = vec3(1, 1, 1);

    for (int bounce = 0; bounce <= depth; bounce++) {
        double nearest_t = INFINITY;
        polygon* nearest_polygon = nullptr;
        vec3 intersection_point;

        for (int i = 0; i < polygons_cnt; i++) {
            vec3 temp_intersection;
            double t;
            if (ray_intersects_polygon(pos, dir, polygons[i], &temp_intersection, &t) && t < nearest_t) {
                nearest_t = t;
                nearest_polygon = &polygons[i];
                intersection_point = temp_intersection;
            }
        }

        if (!nearest_polygon) {
            return make_uchar4(
                (unsigned char)(fmin(color.x, 1.0) * 255),
                (unsigned char)(fmin(color.y, 1.0) * 255),
                (unsigned char)(fmin(color.z, 1.0) * 255),
                255
            );
        }

        vec3 base_color;
        if (nearest_polygon->has_texture) {
            vec3 edge1 = nearest_polygon->b - nearest_polygon->a;
            vec3 edge2 = nearest_polygon->c - nearest_polygon->a;
            vec3 pvec = cross(dir, edge2);
            double det = dot(edge1, pvec);
            vec3 tvec = pos - nearest_polygon->a;
            double u = dot(tvec, pvec) / det;
            vec3 qvec = cross(tvec, edge1);
            double v = dot(dir, qvec) / det;

            // Семплируем цвет из текстуры
            uchar4 tex_color = sample_texture(nearest_polygon->texture, u, v);
            base_color = vec3(tex_color.x / 255.0, tex_color.y / 255.0, tex_color.z / 255.0);
        } else {
            base_color = vec3(
                nearest_polygon->color.x / 255.0,
                nearest_polygon->color.y / 255.0,
                nearest_polygon->color.z / 255.0
            );
        }

        color = color + base_color * 0.1 * attenuation;

        vec3 total_light = vec3(0, 0, 0);

        for (int i = 0; i < lights_cnt; i++) {
            vec3 light_dir = norm(lights[i].position - intersection_point);

            bool in_shadow = is_in_shadow(intersection_point, light_dir, lights[i].position, polygons, polygons_cnt);

            double shadow_factor = in_shadow ? 0.3 : 1.0;

            double diffuse_intensity = max(0.0, dot(nearest_polygon->normal, light_dir)) * shadow_factor;
            vec3 light_color = vec3(
                lights[i].color.x / 255.0,
                lights[i].color.y / 255.0,
                lights[i].color.z / 255.0
            ) * lights[i].intensity;

            total_light = total_light + base_color * light_color * diffuse_intensity;
        }

        color = color + total_light * attenuation;

        if (nearest_polygon->transparency > 0.0) {
            attenuation = attenuation * (1.0 - nearest_polygon->transparency); 
            pos = intersection_point + dir * 1e-2; 
            continue; 
        }

        if (nearest_polygon->reflection > 0.0) {
            vec3 reflected_dir = reflect(norm(dir), nearest_polygon->normal);
            pos = intersection_point + reflected_dir * 1e-2; 
            dir = reflected_dir;
            attenuation = attenuation * nearest_polygon->reflection;
            continue; 
        }

        break;
    }

    return make_uchar4(
        (unsigned char)(fmin(color.x, 1.0) * 255),
        (unsigned char)(fmin(color.y, 1.0) * 255),
        (unsigned char)(fmin(color.z, 1.0) * 255),
        255
    );
}


// рендеринг
__host__ __device__ void render(vec3 camera_pos, vec3 camera_view,
                                int w, int h, double angle, uchar4* data,
                                light_source_t* lights, int lights_cnt,
                                polygon* polygons, int polygons_cnt, int depth) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);

    vec3 bz = norm(camera_view - camera_pos);
    vec3 bx = norm(cross(bz, {0.0, 0.0, 1.0}));
    vec3 by = norm(cross(bx, bz));


    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            vec3 v = vec3(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = ray(camera_pos, norm(dir), lights, lights_cnt, polygons, polygons_cnt, camera_pos, depth);
        }
    }
}


// рендеринг на гпу
__global__ void kernel_render(vec3 camera_pos, vec3 camera_view,
                                int w, int h, double angle, uchar4* data,
                                light_source_t* lights, int lights_cnt,
                                polygon* polygons, int polygons_cnt, int depth) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);

    vec3 bz = norm(camera_view - camera_pos);
    vec3 bx = norm(cross(bz, {0.0, 0.0, 1.0}));
    vec3 by = norm(cross(bx, bz));

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            vec3 v = vec3(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3 dir = mult(bx, by, bz, v);
            uchar4 color = ray(camera_pos, norm(dir), lights, lights_cnt, polygons, polygons_cnt, camera_pos, depth);

            double gamma = 2.2;
            color.x = pow(color.x/255.0, 1.0/gamma)*255.0;
            color.y = pow(color.y/255.0, 1.0/gamma)*255.0;
            color.z = pow(color.z/255.0, 1.0/gamma)*255.0;

            data[(h - 1 - j) * w + i] = color;
        }
    }
}

// сглаживание
__host__ __device__ void ssaa(uchar4* data, uchar4* ssaa_data, int w, int h, int k) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            uint4 tmp = make_uint4(0, 0, 0, 0);
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    uchar4 cur_pixel = data[w * k * (y * k + j) + (x * k + i)];
                    tmp.x += cur_pixel.x;
                    tmp.y += cur_pixel.y;
                    tmp.z += cur_pixel.z;
                }
            }
            int rpp = k * k;
            ssaa_data[y * w + x] = make_uchar4(tmp.x / rpp, tmp.y / rpp, tmp.z / rpp, 255);
        }
    }
}

// сглаживание на гпу
__global__ void kernel_ssaa(uchar4* data, uchar4* ssaa_data, int w, int h, int k) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {
            uint4 tmp = make_uint4(0, 0, 0, 0);
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    uchar4 cur_pixel = data[w * k * (y * k + j) + (x * k + i)];
                    tmp.x += cur_pixel.x;
                    tmp.y += cur_pixel.y;
                    tmp.z += cur_pixel.z;
                }
            }
            int rpp = k * k;
            ssaa_data[y * w + x] = make_uchar4(tmp.x / rpp, tmp.y / rpp, tmp.z / rpp, 255);
        }
    }
}

texture_t load_texture(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open texture file: %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    texture_t texture;
    fread(&texture.width, sizeof(int), 1, file);  
    fread(&texture.height, sizeof(int), 1, file); 

    int pixel_count = texture.width * texture.height;
    texture.data = (uchar4*)malloc(sizeof(uchar4) * pixel_count); 
    fread(texture.data, sizeof(uchar4), pixel_count, file);       

    cudaMalloc(&texture.dev_data, sizeof(uchar4) * pixel_count);
    cudaMemcpy(texture.dev_data, texture.data, sizeof(uchar4) * pixel_count, cudaMemcpyHostToDevice);


    fclose(file);
    return texture;
}


void add_scene_polygons(vec3 a, vec3 b, vec3 c, vec3 d, uchar4 color, polygon* polygons, int begin_id, double transp, double coef_ref) {
    polygons[begin_id] = polygon(a, b, c, color);
    polygons[begin_id + 1] = polygon(a, c, d, color);

    for(int i=0; i<2; ++i){
      polygons[begin_id + i].transparency = transp;
      polygons[begin_id + i].reflection = coef_ref;
    }
}

void add_scene_polygons_with_texture(
    vec3 a, vec3 b, vec3 c, vec3 d,
    texture_t texture, polygon* polygons, int begin_id,
    double transp, double coef_ref) {

    polygons[begin_id] = polygon(a, b, c, make_uchar4(0, 0, 0, 255));  
    polygons[begin_id + 1] = polygon(a, c, d, make_uchar4(0, 0, 0, 255));

    for (int i = 0; i < 2; ++i) {
        polygons[begin_id + i].transparency = transp;
        polygons[begin_id + i].reflection = coef_ref;
        polygons[begin_id + i].has_texture = true;     
        polygons[begin_id + i].texture = texture;      
    }
}


void add_tetrahedron_polygons(vec3 center, uchar4 color, double radius, polygon* polygons, int start_index, double transparency, double reflection) {
    double side_length = radius * sqrt(3.0);
    vec3 vertices[4] = {
        vec3(center.x - side_length / 2.0, 0, center.z - side_length / sqrt(12.0)),
        vec3(center.x, center.y + radius, center.z - side_length / sqrt(12.0)),
        vec3(center.x + side_length / 2.0, 0, center.z - side_length / sqrt(12.0)),
        vec3(center.x, center.y, center.z + radius)
    };

    int face_indices[4][3] = {
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3}
    };

    for (int i = 0; i < 4; i++) {
        polygons[start_index + i] = polygon(vertices[face_indices[i][0]], vertices[face_indices[i][1]], vertices[face_indices[i][2]], color);
        polygons[start_index + i].transparency = transparency;
        polygons[start_index + i].reflection = reflection;
    }
}

void add_hexahedron_polygons(vec3 center, uchar4 color, double radius, polygon* polygons, int start_index, double transparency, double reflection) {
    double side_length = 2.0 * radius / sqrt(3.0);
    vec3 origin = vec3(center.x - side_length / 2.0, center.y - side_length / 2.0, center.z - side_length / 2.0);
    vec3 vertices[8] = {
        origin,
        origin + vec3(0, side_length, 0),
        origin + vec3(side_length, side_length, 0),
        origin + vec3(side_length, 0, 0),
        origin + vec3(0, 0, side_length),
        origin + vec3(0, side_length, side_length),
        origin + vec3(side_length, side_length, side_length),
        origin + vec3(side_length, 0, side_length)
    };

    int face_indices[12][3] = {
        {0, 1, 2}, {2, 3, 0}, // передняя грань
        {6, 7, 3}, {3, 2, 6}, // верхняя грань
        {2, 1, 5}, {5, 6, 2}, // правая грань
        {4, 5, 1}, {1, 0, 4}, // задняя грань
        {3, 7, 4}, {4, 0, 3}, // нижняя грань
        {6, 5, 4}, {4, 7, 6}  // левая грань
    };

    for (int i = 0; i < 12; i++) {
        polygons[start_index + i] = polygon(vertices[face_indices[i][0]], vertices[face_indices[i][1]], vertices[face_indices[i][2]], color);
        polygons[start_index + i].transparency = transparency;
        polygons[start_index + i].reflection = reflection;
    }
}

void add_dodecahedron_polygons(vec3 center, uchar4 color, double radius, polygon* polygons, int start_index, double transparency, double reflection) {
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    double inv_phi = 1.0 / phi;
    vec3 raw_vertices[20] = {
        vec3(-inv_phi, 0, phi), vec3(inv_phi, 0, phi), vec3(-1, 1, 1), vec3(1, 1, 1),
        vec3(1, -1, 1), vec3(-1, -1, 1), vec3(0, -phi, inv_phi), vec3(0, phi, inv_phi),
        vec3(-phi, -inv_phi, 0), vec3(-phi, inv_phi, 0), vec3(phi, inv_phi, 0), vec3(phi, -inv_phi, 0),
        vec3(0, -phi, -inv_phi), vec3(0, phi, -inv_phi), vec3(1, 1, -1), vec3(1, -1, -1),
        vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(inv_phi, 0, -phi), vec3(-inv_phi, 0, -phi)
    };

    for (int i = 0; i < 20; i++) {
        raw_vertices[i] = raw_vertices[i] * (radius / sqrt(3.0)) + center;
    }

    int face_indices[36][3] = {
        {4, 0, 6}, {0, 5, 6}, {0, 4, 1}, {0, 3, 7}, {2, 0, 7}, {0, 1, 3},
        {10, 1, 11}, {3, 1, 10}, {1, 4, 11}, {5, 0, 8}, {0, 2, 9}, {8, 0, 9},
        {5, 8, 16}, {6, 5, 12}, {12, 5, 16}, {4, 12, 15}, {4, 6, 12}, {11, 4, 15},
        {2, 13, 17}, {2, 7, 13}, {9, 2, 17}, {13, 3, 14}, {7, 3, 13}, {3, 10, 14},
        {8, 17, 19}, {16, 8, 19}, {8, 9, 17}, {14, 11, 18}, {11, 15, 18}, {10, 11, 14},
        {12, 19, 18}, {15, 12, 18}, {12, 16, 19}, {19, 13, 18}, {17, 13, 19}, {13, 14, 18}
    };

    for (int i = 0; i < 36; i++) {
        polygons[start_index + i] = polygon(raw_vertices[face_indices[i][0]], raw_vertices[face_indices[i][1]], raw_vertices[face_indices[i][2]], color);
        polygons[start_index + i].transparency = transparency;
        polygons[start_index + i].reflection = reflection;
    }
}


void print_default_inputs() {
    std::cout << "100" << std::endl;                   // Количество кадров
    std::cout << "ress1/%d.data" << std::endl;         // Путь к файлам
    std::cout << "600 600 120" << std::endl << std::endl; // Размеры экрана и угол обзора

    std::cout << "7.0 3.0 0.0     2.0 1.0     2.0 6.0 1.0     0.0 0.0" << std::endl;
    std::cout << "2.0 0.0 0.0     0.5 0.1     1.0 4.0 1.0     0.0 0.0" << std::endl << std::endl;

    std::cout << "3.0 3.0 0.5     1.0 0.0 0.0     1.0" << std::endl;
    std::cout << "0.0 0.0 0.0     0.0 1.0 0.0     1.75" << std::endl;
    std::cout << "-3.0 -3.0 0.0     0.0 0.0 1.0     1.5" << std::endl << std::endl;

    std::cout << "-5.0 -5.0 -1.0     -5.0 5.0 -1.0    5.0 5.0 -1.0    5.0 -5.0 -1.0    1.0 0.9 0.25" << std::endl << std::endl;

    std::cout << "1" << std::endl;                    // Количество источников света
    std::cout << "-20.0 0.0 15.0     0.0 0.6 0.6     0.3" << std::endl << std::endl; // Свет

    std::cout << "0.0 0.0 0.0 0.0" << std::endl;      // Прозрачность (сцена, тетраэдр, гексаэдр, додекаэдр)
    std::cout << "0.0 0.5 0.5 0.5" << std::endl;      // Отражение (сцена, тетраэдр, гексаэдр, додекаэдр)

    std::cout << "4" << std::endl;                    // Параметр сглаживания (SSAA)
    std::cout << "2" << std::endl;                   // Глубина рекурсии
}



int main(int argc, char* argv[]) {
    bool use_gpu = true;
    bool use_default = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--default") {
            use_default = true;
        }
        if (arg == "--cpu") {
            use_gpu = false;
        }
        
    }

    if (use_default) {
        print_default_inputs();
        return 0;
    }

    int frames_number; // кол-во кадров
    char output_path[256]; // куда класть изображения
    int w, h; // размеры экрана
    double angle; // угол обзора
    double r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc; // параметры движения камеры
    double r0n, z0n, phi0n, Arn, Azn, wrn, wzn, wphin, prn, pzn;
    // параметры первого тела (тетраэдра)
    double center1_x, center1_y, center1_z;
    double color1_x, color1_y, color1_z;
    double r1;
    // параметры второго тела (куба)
    double center2_x, center2_y, center2_z;
    double color2_x, color2_y, color2_z;
    double r2;
    // параметры третьего тела (додекаэдра)
    double center3_x, center3_y, center3_z;
    double color3_x, color3_y, color3_z;
    double r3;
    // параметры пола
    double floor1_x, floor1_y, floor1_z, floor2_x, floor2_y, floor2_z;
    double floor3_x, floor3_y, floor3_z, floor4_x, floor4_y, floor4_z;
    double floor_color_x, floor_color_y, floor_color_z;

    // параметры фигур
    double transp_scene, transp_tetr, transp_hexa, transp_dode;
    double refl_scene, refl_tetr, refl_hexa, refl_dode;
    //параметры света
    int n_lights;
    double color_light_x, color_light_y, color_light_z;
    double light_intensity;

    double k; // параметры SSAA

    std::cin >> frames_number;
    std::cin >> output_path;
    std::cin >> w >> h >> angle;
    std::cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc;
    std::cin >> r0n >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;
    std::cin >> center1_x >> center1_y >> center1_z;
    std::cin >> color1_x >> color1_y >> color1_z;
    std::cin >> r1;
    std::cin >> center2_x >> center2_y >> center2_z;
    std::cin >> color2_x >> color2_y >> color2_z;
    std::cin >> r2;
    std::cin >> center3_x >> center3_y >> center3_z;
    std::cin >> color3_x >> color3_y >> color3_z;
    std::cin >> r3;
    std::cin >> floor1_x >> floor1_y >> floor1_z >> floor2_x >> floor2_y >> floor2_z;
    std::cin >> floor3_x >> floor3_y >> floor3_z >> floor4_x >> floor4_y >> floor4_z;
    std::cin >> floor_color_x >> floor_color_y >> floor_color_z;
    std::cin >> n_lights;

    

    // Ввод параметров света
    light_source_t* lights;
    lights = (light_source_t*)malloc(sizeof(light_source_t) * n_lights);

    for (int i = 0; i < n_lights; ++i) {
        std::cin >> lights[i].position.x >> lights[i].position.y >> lights[i].position.z;
        std::cin >> color_light_x >> color_light_y >> color_light_z >> light_intensity;
        uchar4 light_color = make_uchar4(color_light_x * 255, color_light_y * 255, color_light_z * 255, 255);
        lights[i].color = light_color;
        lights[i].intensity = light_intensity;
    }

    // Параметры прозрачности и отражения
    std::cin >> transp_scene >> transp_tetr >> transp_hexa >> transp_dode;
    std::cin >> refl_scene >> refl_tetr >> refl_hexa >> refl_dode;
    // Ввод параметра сглаживания SSAA
    std::cin >> k;


    int depth;

    std::cin >> depth;


    texture_t floor_texture;
    bool use_texture = false;
    std::string texture_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--texture" && i + 1 < argc) {  
            texture_file = argv[i + 1];
            use_texture = true;
            break;
        }
    }

    if (use_texture) {
        floor_texture = load_texture(texture_file.c_str());
    }


    // создаем полигоны для объектов
    polygon polygons[54];
    if (use_texture) {
        add_scene_polygons_with_texture(
            vec3(floor1_x, floor1_y, floor1_z),
            vec3(floor2_x, floor2_y, floor2_z),
            vec3(floor3_x, floor3_y, floor3_z),
            vec3(floor4_x, floor4_y, floor4_z),
            floor_texture, polygons, 0, transp_scene, refl_scene
        );
    } else {
        add_scene_polygons(
            vec3(floor1_x, floor1_y, floor1_z),
            vec3(floor2_x, floor2_y, floor2_z),
            vec3(floor3_x, floor3_y, floor3_z),
            vec3(floor4_x, floor4_y, floor4_z),
            make_uchar4(floor_color_x * 255, floor_color_y * 255, floor_color_z * 255, 255),
            polygons, 0, transp_scene, refl_scene
        );
    }

    add_tetrahedron_polygons(
        vec3(center1_x, center1_y, center1_z),
        make_uchar4(color1_x * 255, color1_y * 255, color1_z * 255, 255),
        r1, polygons, 2, transp_tetr, refl_tetr
    );
    add_hexahedron_polygons(
        vec3(center2_x, center2_y, center2_z),
        make_uchar4(color2_x * 255, color2_y * 255, color2_z * 255, 255),
        r2, polygons, 6, transp_hexa, refl_hexa
    );
    add_dodecahedron_polygons(
        vec3(center3_x, center3_y, center3_z),
        make_uchar4(color3_x * 255, color3_y * 255, color3_z * 255, 255),
        r3, polygons, 18, transp_dode, refl_dode
    );


    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h * k * k);
    uchar4* ssaa_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    uchar4* dev_data;
    uchar4* dev_ssaa_data;
    polygon* dev_polygons;
    light_source_t* dev_lights;
    char buff[256];
    if (use_gpu) { 
        CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h * k * k));
        CSC(cudaMalloc(&dev_ssaa_data, sizeof(uchar4) * w * h));
        CSC(cudaMalloc(&dev_polygons, sizeof(polygon) * 54));
        CSC(cudaMalloc(&dev_lights, sizeof(light_source_t) * n_lights));
        CSC(cudaMemcpy(dev_polygons, polygons, sizeof(polygon) * 54, cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_lights, lights, sizeof(light_source_t) * n_lights, cudaMemcpyHostToDevice));

    }


    for (int frame = 0; frame < frames_number; ++frame) {
        double t = 2 * M_PI * frame / frames_number;
        vec3 camera_pos, camera_view;

        double rc = r0c + Arc * sin(wrc * t + prc);
        double zc = z0c + Azc * sin(wzc * t + pzc);
        double phic = phi0c + wphic * t;

        double rn = r0n + Arn * sin(wrn * t + prn);
        double zn = z0n + Azn * sin(wzn * t + pzn);
        double phin = phi0n + wphin * t;

        camera_pos.x = rc * cos(phic);
        camera_pos.y = rc * sin(phic);
        camera_pos.z = zc;

        camera_view.x = rn * cos(phin);
        camera_view.y = rn * sin(phin);
        camera_view.z = zn;

        cudaEvent_t start, stop;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start));

        if (use_gpu) {
            kernel_render<<<dim3(16, 16), dim3(16, 16)>>>(
                camera_pos, camera_view, w * k, h * k, angle,
                dev_data, dev_lights, n_lights, dev_polygons, 54, depth
            );
            cudaDeviceSynchronize();
            CSC(cudaGetLastError());
            kernel_ssaa<<<dim3(16, 16), dim3(16, 16)>>>(dev_data, dev_ssaa_data, w, h, k);
            cudaDeviceSynchronize();
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(ssaa_data, dev_ssaa_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        } else {
            render(camera_pos, camera_view, w * k, h * k, angle,
                data, lights, n_lights, polygons, 54, depth
            );
            ssaa(data, ssaa_data, w, h, k);
        }
        CSC(cudaEventRecord(stop));
        CSC(cudaEventSynchronize(stop));
        float time;
        CSC(cudaEventElapsedTime(&time, start, stop));
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));

        sprintf(buff, output_path, frame);
        FILE* output_file = fopen(buff, "w");
        fwrite(&w, sizeof(int), 1, output_file);
        fwrite(&h, sizeof(int), 1, output_file);
        fwrite(ssaa_data, sizeof(uchar4), w * h, output_file);
        fclose(output_file);

        std::cout << frame+1 << "\t" << time << "\t" << w * h * k * k << std::endl;
    }

    free(data);
    free(ssaa_data);
    free(lights);
    if (use_gpu) {
        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_ssaa_data));
        CSC(cudaFree(dev_lights));
    }
    if (use_texture) {
        free(floor_texture.data);
    }
    return 0;
}