import os
import sys
import cv2
import pytesseract
from config_parser import create_config, read_config
from ocr_library import functions as srf
import pyperclip
from datetime import datetime
import time

def debug_team_detection(image):
    """
    Debug function to show exact BGR values and team detection for each row
    """
    height, width = image.shape[:2]
    check_x = int(width * 0.1)
    check_y = int(height * 0.5)
    
    pixel_color = image[check_y, check_x]
    b, g, r = pixel_color
    
    # Convert BGR to hex for easier comparison
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    return {
        'bgr': (b, g, r),
        'hex': hex_color,
        'position': (check_x, check_y)
    }

def print_status(message):
    """Print status message with timestamp"""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}")

def wait_for_user():
    """Wait for user input before closing"""
    input("\nPress Enter to exit...")

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def verify_tesseract_files(tesseract_dir):
    """Verify all required Tesseract files are present"""
    required_files = [
        'tesseract.exe',
        'tessdata/eng.traineddata'
    ]
    
    for file in required_files:
        file_path = os.path.join(tesseract_dir, file)
        if not os.path.exists(file_path):
            return False
    return True

def setup_tesseract():
    """Setup Tesseract with proper path handling"""
    try:
        if getattr(sys, 'frozen', False):
            tesseract_dir = os.path.join(sys._MEIPASS, 'Tesseract-OCR')
        else:
            tesseract_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tesseract-OCR')

        print_status("Setting up Tesseract OCR...")
        
        if not os.path.exists(tesseract_dir):
            return False
        
        if not verify_tesseract_files(tesseract_dir):
            return False
        
        tesseract_exe = os.path.join(tesseract_dir, 'tesseract.exe')
        tessdata_dir = os.path.join(tesseract_dir, 'tessdata')
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        os.environ['TESSDATA_PREFIX'] = tessdata_dir
        
        return True
        
    except Exception:
        return False

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def main():
    try:
        print_status("Initializing VALScoreboardTracker...")
        
        if not setup_tesseract():
            print_status("Failed to initialize Tesseract OCR. Please ensure it's properly installed.")
            wait_for_user()
            return

        base_path = get_base_path()
        screenshot_folder = os.path.join(base_path, "screenshots")
        config_path = os.path.join(base_path, "config.ini")
        scoreboard_path = os.path.join(base_path, "scoreboard.csv")

        print_status("Loading configuration...")
        if os.path.exists(config_path):
            config_data = read_config()
            print_status("Configuration loaded successfully")
        else:
            print_status("Creating new configuration file...")
            config_data = create_config()

        maps = config_data['maps']

        if os.path.exists(scoreboard_path):
            os.remove(scoreboard_path)
            print_status("Removed old scoreboard file")

        if os.path.exists(screenshot_folder) and os.path.isdir(screenshot_folder):
            screenshots = [f for f in os.listdir(screenshot_folder) if f.lower().endswith(".png")]
            print_status(f"Found {len(screenshots)} screenshots to process")

            for filename in screenshots:
                file_path = os.path.join(screenshot_folder, filename)
                print_status(f"Processing screenshot: {filename}")

                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image_colored = cv2.imread(file_path)
                
                print_status("Detecting map...")
                map_name = srf.find_map_name(image, maps)
                print_status(f"Map detected: {map_name}")

                print_status("Processing scoreboard table...")
                image, image_colored = srf.find_tables(image, image_colored)

                print_status("Extracting cell information...")
                cell_images_rows, headshots_images_rows = srf.extract_cell_images_from_table(image, image_colored)

                print_status("Identifying agents...")
                agents = srf.identify_agents(headshots_images_rows)

                print_status("Reading table data...")
                output = srf.read_table_rows(cell_images_rows, headshots_images_rows)
                current_date = datetime.now().strftime("%d/%m/%Y")
                
                print_status("Merging data...")
                merged_output = []
                for i, row in enumerate(output):
                    # Extract team information (first element) and rest of the data
                    team = row[0]
                    player_data = row[1:]
                    
                    # Create merged row with date, team, map, agent, and player data
                    merged_row = [current_date, team, map_name, agents[i]] + player_data
                    merged_output.append(merged_row)

                print_status("Filtering team1 data...")
                # Filter only rows where team is 'team1'
                filtered_output = [row for row in merged_output if row[1] == 'team1']

                if not filtered_output:
                    print_status("Warning: No team1 data found in this screenshot")
                    continue

                print_status("Cleaning output format...")
                # Then remove team identifier from filtered data
                clean_output = [[row[0], row[2], row[3]] + row[4:] for row in filtered_output]

                srf.write_csv(clean_output)
                print_status(f"Data written to CSV for {filename}")

        else:
            print_status("No screenshots folder found or no PNG files to process")
            return

        print_status("Cleaning up temporary files...")
        if os.path.exists(os.path.join(base_path, 'temp_headshot.png')):
            os.remove(os.path.join(base_path, 'temp_headshot.png'))
            print_status("Temporary files cleaned up")

        print_status("Reading final scoreboard data...")
        with open(scoreboard_path, "r", encoding="utf-8") as file:
            scoreboard_data = file.read()

        pyperclip.copy(scoreboard_data)
        print_status("Scoreboard data copied to clipboard")
        
        print_status("Processing completed successfully!")

    except Exception as e:
        print_status(f"Error occurred: {str(e)}")
        print_status("Please check the error message above and try again")
    
    finally:
        wait_for_user()

if __name__ == "__main__":
    main()