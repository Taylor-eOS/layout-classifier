import pygame
import fitz

def preload_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    zoom_x = zoom_y = 2
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = page.get_pixmap(matrix=mat)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = pygame.image.fromstring(pix.samples, [pix.width, pix.height], mode)
    return img, pix.width, pix.height

def show_pdf_page_with_block(pdf_path, block, predicted_class, page_number):
    pygame.init()
    try:
        img, img_width, img_height = preload_pdf_page(pdf_path, page_number)
        
        button_space_height = 100
        scale_factor = 1.3
        screen_width = int(img_width // 2 * scale_factor)
        screen_height = int(img_height // 2 * scale_factor) + button_space_height
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Layout Classifier")
        img = pygame.transform.smoothscale(img, (screen_width, screen_height - button_space_height))

        highlight_colors = {
            0: (200, 200, 200),
            1: (160, 160, 160),
            2: (120, 120, 120),
            3: (80, 80, 80)
        }
        button_texts = ["Header", "Body", "Footer", "Quote"]
        pygame.font.init()
        font = pygame.font.Font(None, 24)

        buttons = [
            pygame.Rect(10, screen_height - button_space_height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(80, screen_height - button_space_height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(150, screen_height - button_space_height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(220, screen_height - button_space_height + (button_space_height - 60) // 2, 60, 60)
        ]

        selected_idx = None
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for idx, button in enumerate(buttons):
                        if button.collidepoint(event.pos):
                            selected_idx = idx
                            running = False

            screen.fill((255, 255, 255))
            screen.blit(img, (0, 0))

            x0, y0, x1, y1 = block['x0'] * scale_factor, block['y0'] * scale_factor, block['x1'] * scale_factor, block['y1'] * scale_factor
            pygame.draw.rect(screen, highlight_colors[predicted_class], pygame.Rect(x0, y0, x1 - x0, y1 - y0), 4)

            for idx, button in enumerate(buttons):
                pygame.draw.rect(screen, (230, 230, 230), button)
                text_surface = font.render(button_texts[idx], True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=button.center)
                screen.blit(text_surface, text_rect)

            pygame.display.flip()

        return selected_idx

    finally:
        pygame.quit()

