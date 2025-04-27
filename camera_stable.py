
import pygame
import numpy as np
import math
import sys

# === Funkcje tworzące macierze transformacji (Układ Lewoskrętny - LH) ===
# Podstawowe macierze transformacji geometrycznych 3D

def create_identity_matrix():
    """Tworzy macierz jednostkową 4x4."""
    return np.identity(4, dtype=np.float32)

def create_translation_matrix(tx, ty, tz):
    """Tworzy macierz translacji 4x4."""
    # Standardowa macierz przesunięcia o wektor [tx, ty, tz]
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)

# --- Macierze Rotacji (Lewoskrętne - LH) ---
# Standardowe macierze obrotu wokół osi X, Y, Z w układzie lewoskrętnym (LH)

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
# --- Koniec Macierzy Rotacji LH ---


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

    # Konstrukcja macierzy projekcji perspektywicznej
    # Mapuje objętość widzenia (frustum) do znormalizowanej kostki (NDC)
    # Zachowuje informację o głębokości w sposób nieliniowy (mapuje w = -z_view)
    return np.array([
        [f / aspect_ratio, 0, 0,                            0],
        [0,                f, 0,                            0],
        [0,                0, -(far + near) / z_range,     -(2 * far * near) / z_range],
        [0,                0, -1,                           0] # Kluczowe dla zachowania głębokości i dzielenia perspektywicznego
    ], dtype=np.float32)

# === Funkcja pomocnicza do projekcji ===
def project_to_screen(clip_coords, width, height):
    """
    Przekształca współrzędne z przestrzeni Clip Space (po projekcji)
    na współrzędne ekranu (piksele).
    """
    # Współrzędna 'w' przechowuje informację o głębokości sprzed projekcji
    w = clip_coords[3]
    # Zabezpieczenie przed dzieleniem przez zero (punkty w nieskończoności lub na środku rzutowania)
    if abs(w) < 1e-7: return None

    # Dzielenie perspektywiczne: uzyskanie Znormalizowanych Współrzędnych Urządzenia (NDC)
    # Zakres NDC to zazwyczaj [-1, 1] dla x, y, z.
    ndc_x = clip_coords[0] / w
    ndc_y = clip_coords[1] / w

    # Transformacja Viewport: mapowanie NDC [-1, 1] na współrzędne ekranu [0, Width]x[0, Height]
    screen_x = int((ndc_x + 1) * 0.5 * width)
    # Oś Y w Pygame rośnie w dół, więc odwracamy kierunek osi Y z NDC
    screen_y = int((1 - ndc_y) * 0.5 * height)

    return (screen_x, screen_y)

# === Krok 2: Inicjalizacja Pygame i ustawienia okna ===
pygame.init()  # Inicjalizacja Pygame
pygame.font.init() # Inicjalizacja modułu czcionek

# Ustawienia okna
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wirtualna Kamera (Stacjonarna, LH) - Kwadrat 2x2 + Roll (Near Plane Clipping)")
clock = pygame.time.Clock() # Zegar do kontroli FPS
BLACK = (0, 0, 0); WHITE = (255, 255, 255) # Kolory
BLUE = (0, 0, 255) # Kolor tekstu kontrolek

# Ładowanie czcionki do wyświetlania tekstu
try:
    controls_font = pygame.font.SysFont('consolas', 16)
except:
    controls_font = pygame.font.SysFont(None, 20) # Domyślna, jeśli 'consolas' nie ma

# === Parametry Symulowanej Kamery i Świata ===
# Koncepcja "Stacjonarnej Kamery": Kamera zawsze jest w (0,0,0) układu Widoku.
# Ruch jest symulowany przez PRZECIWNE przesuwanie całego świata.
# `world_offset` przechowuje to odwrotne przesunięcie świata.
world_offset = np.array([0.0, 0.0, 15.0], dtype=np.float32) # Świat przesunięty o +15 na Z (jakby kamera była w -15)

# Kąty określają orientację kamery (Yaw, Pitch, Roll w radianach)
camera_yaw = 0.0    # Obrót wokół osi Y
camera_pitch = 0.0  # Obrót wokół osi X
camera_roll = 0.0   # Obrót wokół osi Z

# Parametry widoku i sterowania
current_fov = 60.0             # Aktualny kąt widzenia kamery (w stopniach)
ROLL_SPEED = math.radians(60)  # Prędkość obrotu Roll (w radianach na sekundę)
MOVE_SPEED = 5.0               # Prędkość "ruchu" kamery (prędkość przesuwania świata)
MOUSE_SENSITIVITY = 0.002     
ZOOM_SPEED = 150.0              # Prędkość zmiany FOV (zoom)
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Wektor wskazujący globalną "górę" świata
PITCH_LIMIT = math.pi / 2 - 0.01 # Maksymalny kąt spojrzenia w górę/dół (nieco mniej niż 90 stopni)

# Ustawienia myszy dla trybu FPS
pygame.mouse.set_visible(False) # Ukrycie kursora systemowego
pygame.event.set_grab(True)   # Zablokowanie kursora w oknie

# === Definicja OBIEKTU PODSTAWOWEGO (sześcian) ===
# Wierzchołki sześcianu zdefiniowane wokół jego lokalnego środka (0,0,0)
base_vertices = np.array([
    # Współrzędne (x, y, z, w) - 'w'=1 dla punktów (współrzędne jednorodne)
    [-1,-1, -1, 1], [ 1,-1, -1, 1], [ 1, 1, -1, 1], [-1, 1, -1, 1], # Ściana tylna (-Z w lokalnym układzie)
    [-1,-1,  1, 1], [ 1,-1,  1, 1], [ 1, 1,  1, 1], [-1, 1,  1, 1]  # Ściana przednia (+Z w lokalnym układzie)
], dtype=np.float32)

# Krawędzie: Pary indeksów wierzchołków, które należy połączyć linią
edges = [
    (0,1),(1,2),(2,3),(3,0), # Krawędzie ściany tylnej
    (4,5),(5,6),(6,7),(7,4), # Krawędzie ściany przedniej
    (0,4),(1,5),(2,6),(3,7)  # Krawędzie łączące ścianę przednią i tylną
]

# === Definicja Sceny (Rozmieszczenie Obiektów) ===
# Lista macierzy translacji (Modelu) dla czterech sześcianów
# Każda macierz przesuwa KOPIĘ sześcianu `base_vertices` w inne miejsce świata.
cube_model_translations = [
    create_translation_matrix(-1.5, 0.0,  1.5), # Sześcian 1: lewo-przód
    create_translation_matrix( 1.5, 0.0,  1.5), # Sześcian 2: prawo-przód
    create_translation_matrix(-1.5, 0.0, -1.5), # Sześcian 3: lewo-tył
    create_translation_matrix( 1.5, 0.0, -1.5)  # Sześcian 4: prawo-tył
]

# Parametry Piramidy Widzenia (View Frustum)
Z_NEAR = 0.1           # Odległość do bliskiej płaszczyzny przycinania
Z_FAR = 100.0          # Odległość do dalekiej płaszczyzny przycinania
ASPECT_RATIO = WIDTH / HEIGHT # Stosunek szerokości do wysokości okna

# === Krok 3: Główna pętla programu ===
running = True
while running:
    # === Krok 3a: Czas i obsługa zdarzeń ===
    # Obliczenie czasu, jaki upłynął od ostatniej klatki (dla płynności ruchu)
    dt = clock.tick(60) / 1000.0
    fov_change = 0.0 # Zmiana FOV w tej klatce

    # Pętla obsługi zdarzeń (wciśnięcia klawiszy, ruch myszy, etc.)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False # Zamknięcie okna
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False # Klawisz ESC
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Obsługa zoomu kółkiem myszy
            if event.button == 4: fov_change = -ZOOM_SPEED * dt # Przybliżenie (mniejsze FOV)
            elif event.button == 5: fov_change = ZOOM_SPEED * dt # Oddalenie (większe FOV)

    # === Krok 3b: Aktualizacja Orientacji "Kamery" i Zoomu ===
    dx, dy = pygame.mouse.get_rel() # Odczyt względnego ruchu myszy od ostatniej klatki
    keys = pygame.key.get_pressed() # Odczyt stanu wszystkich klawiszy

    # dodatkowo odczyt z przyciskow strzalek
    if keys[pygame.K_UP] : dy =- 5
    if keys[pygame.K_LEFT] : dx = -5
    if keys[pygame.K_RIGHT] : dx = 5
    if keys[pygame.K_DOWN]  : dy = 5

    # Aktualizacja kątów orientacji kamery na podstawie ruchu myszy
    camera_yaw += dx * MOUSE_SENSITIVITY
    camera_pitch += dy * MOUSE_SENSITIVITY
    # Ograniczenie kąta Pitch, aby uniknąć "przewrotki"
    camera_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, camera_pitch))

    # Aktualizacja FOV (zoom) na podstawie klawiszy +/- lub kółka myszy
    if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]: fov_change -= ZOOM_SPEED * dt
    if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: fov_change += ZOOM_SPEED * dt
    current_fov += fov_change
    # Ograniczenie FOV do rozsądnego zakresu
    current_fov = max(1.0, min(170.0, current_fov))

    # Aktualizacja kąta Roll (przechylenia) kamery klawiszami Z/X
    if keys[pygame.K_z]: camera_roll += ROLL_SPEED * dt
    if keys[pygame.K_x]: camera_roll -= ROLL_SPEED * dt

    # === Obliczanie Aktualnych Wektorów Kierunkowych Kamery ===
    # Obliczanie macierzy rotacji dla każdej osi na podstawie aktualnych kątów
    cam_rot_y = rotate_y_lh(camera_yaw)
    cam_rot_x = rotate_x_lh(camera_pitch)
    cam_rot_z = rotate_z_lh(camera_roll)

    # Połączenie rotacji w jedną macierz orientacji kamery
    # Kolejność mnożenia (Yaw -> Pitch -> Roll) definiuje sposób obrotu
    camera_orientation_matrix = cam_rot_y @ cam_rot_x @ cam_rot_z

    # Definicja podstawowych wektorów w lokalnym układzie kamery (osie kamery)
    local_forward = np.array([0, 0, 1, 0]) # Lokalna oś Z (do przodu w LH)
    local_right   = np.array([1, 0, 0, 0]) # Lokalna oś X (w prawo w LH)
    local_up      = np.array([0, 1, 0, 0]) # Lokalna oś Y (w górę w LH)

    # Transformacja wektorów lokalnych do układu świata za pomocą macierzy orientacji
    world_forward_vec = camera_orientation_matrix @ local_forward
    world_right_vec   = camera_orientation_matrix @ local_right
    world_up_vec      = camera_orientation_matrix @ local_up # Wektor "góry" kamery w świecie

    # Wyciągnięcie składowych XYZ i normalizacja (uzyskanie wektorów jednostkowych)
    camera_front = world_forward_vec[:3]
    camera_right = world_right_vec[:3]
    camera_up    = world_up_vec[:3] # Rzeczywisty kierunek "góry" kamery

    # Normalizacja wektorów kierunkowych (zapewnienie długości 1)
    norm_front = np.linalg.norm(camera_front)
    norm_right = np.linalg.norm(camera_right)
    norm_up    = np.linalg.norm(camera_up)

    # Zabezpieczenie przed dzieleniem przez zero, jeśli wektor jest zerowy
    if norm_front > 1e-6: camera_front /= norm_front
    else: camera_front = np.array([0., 0., 1.]) # Domyślny kierunek

    if norm_right > 1e-6: camera_right /= norm_right
    else: camera_right = np.array([1., 0., 0.]) # Domyślny kierunek

    if norm_up > 1e-6: camera_up /= norm_up
    else: camera_up = np.array([0., 1., 0.]) # Domyślny kierunek

    # === Aktualizacja Przesunięcia Świata (Symulacja Ruchu Kamery) ===
    # Obliczenie wektora zamierzonego ruchu na podstawie wciśniętych klawiszy
    move_direction = np.array([0.,0.,0.], dtype=np.float32)
    if keys[pygame.K_w]: move_direction += camera_front  # Do przodu (względem kamery)
    if keys[pygame.K_s]: move_direction -= camera_front  # Do tyłu (względem kamery)
    if keys[pygame.K_d]: move_direction -= camera_right  # W prawo (względem kamery)
    if keys[pygame.K_a]: move_direction += camera_right  # W lewo (względem kamery)
    if keys[pygame.K_SPACE] or keys[pygame.K_e]: move_direction -= WORLD_UP # W górę (względem świata)
    if keys[pygame.K_LSHIFT] or keys[pygame.K_q]: move_direction += WORLD_UP # W dół (względem świata)

    # Normalizacja wektora ruchu (jeśli jakikolwiek ruch był zamierzony)
    move_vec_norm = np.linalg.norm(move_direction)
    if move_vec_norm > 1e-6:
        move_direction = move_direction / move_vec_norm

    # Obliczenie wektora przesunięcia w tej klatce (prędkość * czas)
    delta_camera_move = move_direction * MOVE_SPEED * dt

    # Aktualizacja przesunięcia świata: świat przesuwa się w PRZECIWNYM kierunku
    world_offset -= delta_camera_move

    # === Krok 3c: Obliczanie Macierzy Widoku i Projekcji ===

    # 1. Macierz Translacji Świata: Przesuwa świat o obliczony `world_offset`
    inv_translation = create_translation_matrix(world_offset[0], world_offset[1], world_offset[2])

    # 2. Macierze Odwrotnych Rotacji: Obracają świat przeciwnie do orientacji kamery
    rot_z_inv = rotate_z_lh(-camera_roll)
    rot_x_inv = rotate_x_lh(-camera_pitch)
    rot_y_inv = rotate_y_lh(-camera_yaw)

    # 3. Finalna Macierz Widoku (View Matrix): Łączy odwrotne rotacje i translacje.
    # Transformuje współrzędne ze świata do układu współrzędnych kamery.
    # Kolejność: V = InvRot * InvTrans
    view_matrix = rot_z_inv @ rot_x_inv @ rot_y_inv @ inv_translation

    # 4. Macierz Projekcji (Projection Matrix): Definiuje perspektywę.
    projection_matrix = create_projection_matrix_lh(current_fov, ASPECT_RATIO, Z_NEAR, Z_FAR)

    # === Krok 3d: Rysowanie Sceny ===
    # Wypełnienie tła ekranu kolorem czarnym (czyszczenie poprzedniej klatki)
    screen.fill(BLACK)

    # Wyświetlanie tekstu z informacjami o sterowaniu
    controls_text = [
        "Sterowanie:", "WASD: Ruch Poziomy", "QE/Spacja/Shift: Ruch Pionowy",
        "Mysz / Klawisze strzałek : Rozglądanie (Yaw/Pitch)", "Kółko / +/-: Zoom (FOV)",
        "Z/X: Obrót (Roll)", "ESC: Wyjście"
    ]
    text_y = 10
    for line in controls_text:
        text_surface = controls_font.render(line, True, BLUE)
        screen.blit(text_surface, (10, text_y))
        text_y += text_surface.get_height() + 2

    # --- Pętla Rysowania Obiektów ---
    # Iteracja przez macierze Modelu dla każdego sześcianu
    for model_matrix in cube_model_translations:
        # Obliczenie pełnej macierzy Model-View-Projection dla tego konkretnego sześcianu
        mvp_matrix = projection_matrix @ view_matrix @ model_matrix

        # --- Pętla Rysowania Krawędzi ---
        # Iteracja przez wszystkie zdefiniowane krawędzie sześcianu
        for edge in edges:
            idx1, idx2 = edge # Indeksy wierzchołków tworzących krawędź
            # Pobranie współrzędnych wierzchołków w przestrzeni modelu (lokalnej sześcianu)
            v1_model = base_vertices[idx1]
            v2_model = base_vertices[idx2]

            # Transformacja wierzchołków do przestrzeni Clip Space za pomocą macierzy MVP
            clip1 = mvp_matrix @ v1_model
            clip2 = mvp_matrix @ v2_model

            # --- Near Plane Clipping ---
            # Sprawdzenie widoczności punktów względem bliskiej płaszczyzny przycinania (Z_NEAR)
            # Wykorzystujemy współrzędną 'w' (clip[3]), która dla tej macierzy projekcji jest powiązana z -z_view.
            epsilon = 1e-6 # Mały margines błędu dla liczb zmiennoprzecinkowych
            # Warunek visible = (w <= -Z_NEAR) oznacza, że punkt jest przed lub na płaszczyźnie near
            visible1 = clip1[3] <= -Z_NEAR + epsilon
            visible2 = clip2[3] <= -Z_NEAR + epsilon

            p1_screen, p2_screen = None, None # Zmienne na współrzędne ekranowe

            if visible1 and visible2:
                # Przypadek 1: Oba punkty krawędzi są widoczne
                p1_screen = project_to_screen(clip1, WIDTH, HEIGHT)
                p2_screen = project_to_screen(clip2, WIDTH, HEIGHT)
            elif visible1 != visible2:
                # Przypadek 2: Jeden punkt widoczny, drugi nie - trzeba przyciąć krawędź
                # Upewniamy się, że clip1 jest zawsze widoczny (zamieniamy jeśli trzeba)
                if not visible1:
                    clip1, clip2 = clip2, clip1 # Zamiana punktów w Clip Space

                # Obliczenie parametru interpolacji 't' dla punktu przecięcia z płaszczyzną Near
                w1 = clip1[3]
                w2 = clip2[3]
                denominator = w1 - w2
                if abs(denominator) > epsilon: # Unikamy dzielenia przez zero
                    # t = (w1 - target_w) / (w1 - w2), gdzie target_w = -Z_NEAR
                    t = (w1 - (-Z_NEAR)) / denominator
                    t = max(0.0, min(1.0, t)) # Ograniczenie t do zakresu [0, 1]

                    # Obliczenie współrzędnych punktu przecięcia przez interpolację liniową w Clip Space
                    clip_intersection = clip1 + t * (clip2 - clip1)
                    clip_intersection[3] = -Z_NEAR # Upewnienie się co do wartości 'w'

                    # Rzutowanie widocznego punktu i punktu przecięcia na ekran
                    p1_screen = project_to_screen(clip1, WIDTH, HEIGHT)
                    p2_screen = project_to_screen(clip_intersection, WIDTH, HEIGHT)
            # else: Przypadek 3: Oba punkty niewidoczne - nic nie rysujemy

            # --- Rysowanie Linii ---
            # Jeśli uzyskano poprawne współrzędne ekranowe (po projekcji i ew. clippingu)
            if p1_screen and p2_screen:
                pygame.draw.line(screen, WHITE, p1_screen, p2_screen, 1) # Rysuj białą linię

    # === Krok 3e: Aktualizacja wyświetlacza ===
    # Pokazanie gotowej klatki na ekranie
    pygame.display.flip()

# === Krok 4: Zakończenie pracy ===
# Zwolnienie zasobów Pygame i zakończenie programu
pygame.quit()
pygame.font.quit()
sys.exit()