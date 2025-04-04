import logging
from config.appConfig import AppConfig

# Initialize AppConfig (this will initialize the LLMManager with the default model)
app_config = AppConfig()
logging.info("Global app_config initialized")
