# train.py
#
# This is the top-level training entry point.
# It simply imports and calls src/trainer.py:main()

def main():
    # Import trainer AFTER modifying Python path
    from src.trainer import main as trainer_main

    # Call training entry point
    trainer_main()

if __name__ == "__main__":
    main()
