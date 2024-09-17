import pygame
import fitz

def preload_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    mat = fitz.Identity
    pix = page.get_pixmap(matrix=mat)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = pygame.image.fromstring(pix.samples, [pix.width, pix.height], mode)
    return img, pix.width, pix.height

def show_pdf_page_with_block(pdf_path, block, predicted_class, page_number):
    pygame.init()
    try:
        img, img_width, img_height = preload_pdf_page(pdf_path, page_number)
        
        button_space_height = 100
        max_width = 800
        scale_factor = min(1.0, max_width / img_width)
        img_width = int(img_width * scale_factor)
        img_height = int(img_height * scale_factor)
        screen_width = img_width
        screen_height = img_height + button_space_height
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Layout Classifier")
        img = pygame.transform.smoothscale(img, (img_width, img_height))

        highlight_colors = {
            0: (200, 200, 200),
            1: (160, 160, 160),
            2: (120, 120, 120),
            3: (80, 80, 80)
        }
        button_texts = ["Header", "Body", "Footer", "Quote"]
        font = pygame.font.SysFont(None, 24)

        buttons = [
            pygame.Rect(10, img_height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(80, img_height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(150, img_height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(220, img_height + (button_space_height - 60) // 2, 60, 60)]

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

            x0, y0, x1, y1 = [coord * scale_factor for coord in (block['x0'], block['y0'], block['x1'], block['y1'])]
            pygame.draw.rect(screen, highlight_colors.get(predicted_class, (0, 0, 0)), pygame.Rect(x0, y0, x1 - x0, y1 - y0), 4)

            for idx, button in enumerate(buttons):
                pygame.draw.rect(screen, (230, 230, 230), button)
                text_surface = font.render(button_texts[idx], True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=button.center)
                screen.blit(text_surface, text_rect)

            pygame.display.flip()

        return selected_idx

    finally:
        pygame.quit()

