import pygame
import fitz  # PyMuPDF

def show_pdf_page_with_block(pdf_path, block, predicted_class, page_number):
    pygame.init()
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        pix = page.get_pixmap()
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = pygame.image.fromstring(pix.samples, [pix.width, pix.height], mode)

        button_space_height = 100
        screen_height = pix.height + button_space_height
        screen = pygame.display.set_mode((pix.width, screen_height))
        pygame.display.set_caption("PDF Viewer with Annotations")

        # Different shades of grey for rectangle highlights
        highlight_colors = {
            0: (200, 200, 200),  # Lighter grey for header
            1: (160, 160, 160),  # Medium grey for body
            2: (120, 120, 120),  # Darker grey for footer
            3: (80, 80, 80)      # Darkest grey for neither
        }
        button_texts = ["Header", "Body", "Footer", "Neither"]
        pygame.font.init()
        font = pygame.font.Font(None, 24)

        buttons = [
            pygame.Rect(10, pix.height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(80, pix.height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(150, pix.height + (button_space_height - 60) // 2, 60, 60),
            pygame.Rect(220, pix.height + (button_space_height - 60) // 2, 60, 60)
        ]

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return None

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for idx, button in enumerate(buttons):
                        if button.collidepoint(event.pos):
                            pygame.quit()  # Quit pygame before returning
                            return idx  # Return the index corresponding to the button pressed

            screen.fill((255, 255, 255))
            screen.blit(img, (0, 0))
            x0, y0, x1, y1 = block['x0'], block['y0'], block['x1'], block['y1']
            # Draw the rectangle around the block in the grey tone corresponding to the predicted class
            pygame.draw.rect(screen, highlight_colors[predicted_class], pygame.Rect(x0, y0, x1 - x0, y1 - y0), 4)

            for idx, button in enumerate(buttons):
                pygame.draw.rect(screen, (230, 230, 230), button)  # Light grey buttons
                text_surface = font.render(button_texts[idx], True, (0, 0, 0))  # Black text for visibility
                text_rect = text_surface.get_rect(center=button.center)
                screen.blit(text_surface, text_rect)

            pygame.display.flip()

    finally:
        pygame.quit()  # Ensure pygame quits even if an error occurs

