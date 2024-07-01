import json
import logging
import os


LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
LOG_FILE = 'div4aep.log'
CONFIG_FILE = 'config.json'


logger = logging.getLogger(__name__)


def main():

    global logger

    # set up logging to console
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    del handler
    logger.setLevel(logging.INFO)
    logger.info('Set log level to INFO.')
    
    # load config file
    logger.info(f'Reading config file {CONFIG_FILE}.')
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        del f
    except Exception as e:
        logger.error(f'Could not load config file config.json ({e}).')
        return

    # set up logging to log file
    path = os.path.join(config.get('logs_path', ''), LOG_FILE)
    logger.info(f'Starting logging to file {path}.')
    try:
        handler = logging.FileHandler(path, mode='w')
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        del handler
    except Exception as e:
        logger.error(f'Creating log file failed ({e}).')
    del path

    # set log level
    if config.get('debug', False):
        logger.setLevel(logging.DEBUG)
        logger.info('Set log level to DEBUG.')

    # todo


    # clean up
    del logger


if __name__ == '__main__':
    main()
