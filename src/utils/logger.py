import logging


def get_logger(out_file: str) -> logging.Logger:
    logging.basicConfig(filename=out_file,
                        filemode="a",
                        level=logging.INFO,
                        format="%(asctime)s - %(message)s")

    logger = logging.getLogger('python-logstash-logger')
    logger.setLevel(logging.INFO)
    return logger
