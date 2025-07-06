from project2.config import Config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config1")
    args = parser.parse_args()
    try:
        config = Config(args.config)
        print(config.get_config_data())
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
