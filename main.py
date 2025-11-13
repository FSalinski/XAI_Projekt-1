'''
Main module for the project
'''

import logging

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Main module started")
    logging.info("=" * 50)
    # TO DO

if __name__ == "__main__":
    main()
