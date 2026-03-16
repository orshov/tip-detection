# Example Usage of TipDetector

from tip_detector import TipDetector

# Initialize the TipDetector

def main():
    detector = TipDetector()
    
    # Example: Assuming you have a list of prices
    prices = [100, 101, 99, 95, 102]
    
    # Detect tips
    tips = detector.detect(prices)
    
    print("Detected Tips:", tips)

if __name__ == '__main__':
    main()