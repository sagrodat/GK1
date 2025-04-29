import pygame
import numpy as np
import math
import sys

# === Funkcje tworzące macierze transformacji (Układ Lewoskrętny - LH) ===

def create_identity_matrix():
    """Tworzy macierz jednostkową 4x4."""
    return np.identity(4, dtype=np.float32)

def create_translation_matrix(tx, ty, tz):
    """Tworzy macierz translacji 4x4."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)

# --- Macierze Rotacji (Lewoskrętne - LH) ---

def rotate_x_lh(angle_rad):
    """Zwraca macierz rotacji wokół osi X (Pitch) w układzie LH."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1, 0,  0, 0],
        [0, c,  s, 0],
        [0,-s,  c, 0],
        [0, 0,  0, 1]
    ], dtype=np.float32)

def rotate_y_lh(angle_rad):
    """Zwraca macierz rotacji wokół osi Y (Yaw) w układzie LH."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, 0, -s, 0],
        [0, 1,  0, 0],
        [s, 0,  c, 0],
        [0, 0,  0, 1]
    ], dtype=np.float32)

def rotate_z_lh(angle_rad):
    """Zwraca macierz rotacji wokół osi Z (Roll) w układzie LH."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [ c, s, 0, 0],
        [-s, c, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float32)

def create_projection_matrix_lh(fov_deg, aspect_ratio, near, far):
    """
    Tworzy macierz projekcji perspektywicznej dla układu LH.
    """
    fov_rad = math.radians(fov_deg)
    # Obliczenie skali 'f' na podstawie pola widzenia (FOV)
    # f = 1 / tan(fov / 2) = cotan(fov / 2)
    if math.tan(fov_rad / 2.0) == 0: f = float('inf')
    else: f = 1.0 / math.tan(fov_rad / 2.0)

    # Uniknięcie dzielenia przez zero, gdy płaszczyzny Near i Far są w tym samym miejscu
    if far == near: return np.identity(4, dtype=np.float32)
    z_range = far - near

    # Mapuje objętość widzenia (frustum) do znormalizowanej kostki (NDC)
    # Zachowuje informację o głębokości w sposób nieliniowy (mapuje w = -z_view)
    return np.array([
        [f / aspect_ratio, 0, 0,                       0],
        [0,                f, 0,                       0],
        [0,                0, -(far + near) / z_range,   -(2 * far * near) / z_range],
        [0,                0, -1,                      0] # Kluczowe dla zachowania głębokości i dzielenia perspektywicznego
    ], dtype=np.float32)

def project_to_screen(clip_coords, width, height):
    """
    Przekształca współrzędne z przestrzeni Clip Space (po projekcji)
    na współrzędne ekranu (piksele).
    """
    # Współrzędna 'w' przechowuje informację o głębokości sprzed projekcji
    w = clip_coords[3]
    # Zabezpieczenie przed dzieleniem przez zero
    if abs(w) < 1e-7: return None

    # Dzielenie perspektywiczne: uzyskanie Znormalizowanych Współrzędnych Urządzenia (NDC)
    ndc_x = clip_coords[0] / w
    ndc_y = clip_coords[1] / w

    # Transformacja Viewport: mapowanie NDC [-1, 1] na współrzędne ekranu [0, Width]x[0, Height]
    screen_x = int((ndc_x + 1) * 0.5 * width)
    # Oś Y w Pygame rośnie w dół, więc odwracamy kierunek osi Y z NDC
    screen_y = int((1 - ndc_y) * 0.5 * height)

    return (screen_x, screen_y)

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 1280, 960
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wirtualna Kamera (Stacjonarna, LH) - Kwadrat 2x2 + Roll (Near Plane Clipping)")
clock = pygame.time.Clock()
BLACK = (0, 0, 0); WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

try:
    controls_font = pygame.font.SysFont('consolas', 16)
except:
    controls_font = pygame.font.SysFont(None, 20) # Domyślna, jeśli 'consolas' nie ma

# Koncepcja "Stacjonarnej Kamery": Kamera zawsze jest w (0,0,0) układu Widoku.
# Ruch jest symulowany przez PRZECIWNE przesuwanie całego świata.
world_offset = np.array([0.0, 0.0, 15.0], dtype=np.float32) # Świat przesunięty o +15 na Z (jakby kamera była w -15)

camera_yaw = 0.0
camera_pitch = 0.0
camera_roll = 0.0

current_fov = 60.0
ROLL_SPEED = math.radians(60)
MOVE_SPEED = 5.0
MOUSE_SENSITIVITY = 0.002
ZOOM_SPEED = 150.0
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Wektor wskazujący globalną "górę" świata
PITCH_LIMIT = math.pi / 2 - 0.01 # Maksymalny kąt spojrzenia w górę/dół

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

# Wierzchołki sześcianu zdefiniowane wokół jego lokalnego środka (0,0,0)
base_vertices = np.array([
    # Współrzędne (x, y, z, w) - 'w'=1 dla punktów
    [-1,-1, -1, 1], [ 1,-1, -1, 1], [ 1, 1, -1, 1], [-1, 1, -1, 1], # Ściana tylna (-Z w lokalnym układzie)
    [-1,-1,  1, 1], [ 1,-1,  1, 1], [ 1, 1,  1, 1], [-1, 1,  1, 1]  # Ściana przednia (+Z w lokalnym układzie)
], dtype=np.float32)

# Krawędzie: Pary indeksów wierzchołków
edges = [
    (0,1),(1,2),(2,3),(3,0), # tył
    (4,5),(5,6),(6,7),(7,4), # przód
    (0,4),(1,5),(2,6),(3,7)  # łączące
]

# Macierze Modelu dla czterech sześcianów
cube_model_translations = [
    create_translation_matrix(-1.5, 0.0,  1.5), # lewo-przód
    create_translation_matrix( 1.5, 0.0,  1.5), # prawo-przód
    create_translation_matrix(-1.5, 0.0, -1.5), # lewo-tył
    create_translation_matrix( 1.5, 0.0, -1.5)  # prawo-tył
]

Z_NEAR = 0.1            # Bliska płaszczyzna przycinania
Z_FAR = 100.0           # Daleka płaszczyzna przycinania
ASPECT_RATIO = WIDTH / HEIGHT

running = True
while running:
    dt = clock.tick(60) / 1000.0
    fov_change = 0.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False

    keys = pygame.key.get_pressed()

    dx,dy = 0,0

    if keys[pygame.K_UP] : dy =- 5
    if keys[pygame.K_LEFT] : dx = -5
    if keys[pygame.K_RIGHT] : dx = 5
    if keys[pygame.K_DOWN]  : dy = 5

    camera_yaw += dx * MOUSE_SENSITIVITY
    camera_pitch += dy * MOUSE_SENSITIVITY
    # Ograniczenie kąta Pitch
    camera_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, camera_pitch))

    if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]: fov_change -= ZOOM_SPEED * dt
    if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: fov_change += ZOOM_SPEED * dt
    current_fov += fov_change
    # Ograniczenie FOV
    current_fov = max(1.0, min(170.0, current_fov))

    if keys[pygame.K_z]: camera_roll += ROLL_SPEED * dt
    if keys[pygame.K_x]: camera_roll -= ROLL_SPEED * dt

    # --- Obliczanie Wektorów Kierunkowych Kamery ---
    cam_rot_y = rotate_y_lh(camera_yaw)
    cam_rot_x = rotate_x_lh(camera_pitch)
    cam_rot_z = rotate_z_lh(camera_roll)

    # Kolejność mnożenia (Yaw -> Pitch -> Roll)
    camera_orientation_matrix = cam_rot_y @ cam_rot_x @ cam_rot_z

    local_forward = np.array([0, 0, 1, 0])
    local_right   = np.array([1, 0, 0, 0])
    local_up      = np.array([0, 1, 0, 0])

    world_forward_vec = camera_orientation_matrix @ local_forward
    world_right_vec   = camera_orientation_matrix @ local_right
    world_up_vec      = camera_orientation_matrix @ local_up

    camera_front = world_forward_vec[:3]
    camera_right = world_right_vec[:3]
    camera_up    = world_up_vec[:3]

    # Normalizacja wektorów
    norm_front = np.linalg.norm(camera_front)
    norm_right = np.linalg.norm(camera_right)
    norm_up    = np.linalg.norm(camera_up)

    if norm_front > 1e-6: camera_front /= norm_front
    else: camera_front = np.array([0., 0., 1.])

    if norm_right > 1e-6: camera_right /= norm_right
    else: camera_right = np.array([1., 0., 0.])

    if norm_up > 1e-6: camera_up /= norm_up
    else: camera_up = np.array([0., 1., 0.])

    # --- Aktualizacja Przesunięcia Świata (Symulacja Ruchu Kamery) ---
    move_direction = np.array([0.,0.,0.], dtype=np.float32)
    if keys[pygame.K_w]: move_direction += camera_front
    if keys[pygame.K_s]: move_direction -= camera_front
    if keys[pygame.K_d]: move_direction -= camera_right
    if keys[pygame.K_a]: move_direction += camera_right
    if keys[pygame.K_SPACE] or keys[pygame.K_e]: move_direction -= WORLD_UP # W górę (względem świata)
    if keys[pygame.K_LSHIFT] or keys[pygame.K_q]: move_direction += WORLD_UP # W dół (względem świata)

    move_vec_norm = np.linalg.norm(move_direction)
    if move_vec_norm > 1e-6:
        move_direction = move_direction / move_vec_norm

    delta_camera_move = move_direction * MOVE_SPEED * dt
    # Aktualizacja przesunięcia świata: świat przesuwa się w PRZECIWNYM kierunku
    world_offset -= delta_camera_move

    # --- Obliczanie Macierzy Widoku i Projekcji ---
    inv_translation = create_translation_matrix(world_offset[0], world_offset[1], world_offset[2])

    rot_z_inv = rotate_z_lh(-camera_roll)
    rot_x_inv = rotate_x_lh(-camera_pitch)
    rot_y_inv = rotate_y_lh(-camera_yaw)

    # Finalna Macierz Widoku (View Matrix)
    # Kolejność: V = InvRot * InvTrans
    view_matrix = rot_z_inv @ rot_x_inv @ rot_y_inv @ inv_translation

    projection_matrix = create_projection_matrix_lh(current_fov, ASPECT_RATIO, Z_NEAR, Z_FAR)

    # --- Rysowanie ---
    screen.fill(BLACK)

    # Wyświetlanie tekstu sterowania
    controls_text = [
        "Sterowanie:", "WASD: Ruch Poziomy", "QE/Spacja/Shift: Ruch Pionowy",
        "Mysz / Strzałki : Rozglądanie", "+/-: Zoom (FOV)",
        "Z/X: Obrót (Roll)", "ESC: Wyjście"
    ]
    text_y = 10
    for line in controls_text:
        text_surface = controls_font.render(line, True, BLUE)
        screen.blit(text_surface, (10, text_y))
        text_y += text_surface.get_height() + 2

    # Pętla Rysowania Obiektów
    for model_matrix in cube_model_translations:
        mvp_matrix = projection_matrix @ view_matrix @ model_matrix

        # Pętla Rysowania Krawędzi
        for edge in edges:
            idx1, idx2 = edge
            v1_model = base_vertices[idx1]
            v2_model = base_vertices[idx2]

            clip1 = mvp_matrix @ v1_model
            clip2 = mvp_matrix @ v2_model

            # --- Near Plane Clipping ---
            # Sprawdzenie widoczności względem Z_NEAR używając współrzędnej 'w' (clip[3])
            epsilon = 1e-6 # Margines błędu
            # Warunek: punkt jest przed lub na płaszczyźnie near (w <= -Z_NEAR)
            visible1 = clip1[3] <= -Z_NEAR + epsilon
            visible2 = clip2[3] <= -Z_NEAR + epsilon

            p1_screen, p2_screen = None, None

            if visible1 and visible2:
                # Oba punkty widoczne
                p1_screen = project_to_screen(clip1, WIDTH, HEIGHT)
                p2_screen = project_to_screen(clip2, WIDTH, HEIGHT)
            elif visible1 != visible2:
                # Jeden punkt widoczny, drugi nie - przycinamy krawędź
                # Upewniamy się, że clip1 jest zawsze widoczny
                if not visible1:
                    clip1, clip2 = clip2, clip1 # Zamiana

                # Obliczenie parametru interpolacji 't' dla punktu przecięcia z płaszczyzną Near
                w1 = clip1[3]
                w2 = clip2[3]
                denominator = w1 - w2
                if abs(denominator) > epsilon: # Unikamy dzielenia przez zero
                    # t = (w1 - target_w) / (w1 - w2), gdzie target_w = -Z_NEAR
                    t = (w1 - (-Z_NEAR)) / denominator
                    t = max(0.0, min(1.0, t)) # Ograniczenie t do [0, 1]

                    # Interpolacja liniowa w Clip Space
                    clip_intersection = clip1 + t * (clip2 - clip1)
                    clip_intersection[3] = -Z_NEAR # Upewnienie się co do 'w'

                    # Rzutowanie widocznego punktu i punktu przecięcia
                    p1_screen = project_to_screen(clip1, WIDTH, HEIGHT)
                    p2_screen = project_to_screen(clip_intersection, WIDTH, HEIGHT)
            # else: Oba punkty niewidoczne - nic nie rysujemy

            # Rysowanie Linii
            if p1_screen and p2_screen:
                pygame.draw.line(screen, WHITE, p1_screen, p2_screen, 1)

    pygame.display.flip()

pygame.quit()
pygame.font.quit()
sys.exit()