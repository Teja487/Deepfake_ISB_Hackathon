"""
Django settings for project_settings project.
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build paths inside the project like this: os.path.join(PROJECT_DIR, ...)
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '@)0qp0!&-vht7k0wyuihr+nk-b8zrvb5j^1d@vl84cd1%)f=dz'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Change and set this to correct IP/Domain
ALLOWED_HOSTS = ["*"]


# Application definition

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'ml_app.apps.MlAppConfig'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project_settings.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(PROJECT_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media'
            ],
        },
    },
]

WSGI_APPLICATION = 'project_settings.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(PROJECT_DIR, 'db.sqlite3'),
    }
}


# Internationalization
# https://docs.djangoproject.com/en/3.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = False

USE_L10N = False

USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.0/howto/static-files/

#used in production to serve static files
STATIC_ROOT = "/home/app/staticfiles/"
ALLOWED_IMAGE_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'])
ALLOWED_TEXT_EXTENSIONS = ['txt','pdf','docx'] 
#url for static files
STATIC_URL = '/static/'

STATICFILES_DIRS = [
    os.path.join(PROJECT_DIR, 'uploaded_images'),
    os.path.join(PROJECT_DIR, 'images_uploaded'),
    os.path.join(PROJECT_DIR, 'static'),
    os.path.join(PROJECT_DIR, 'models'),
]

#CONTENT_TYPES = ['video']
CONTENT_TYPES = ['video/mp4', 'video/mpeg', 'image/jpeg', 'image/png', 'image/gif']
MAX_UPLOAD_SIZE = "104857600"

MEDIA_URL = "/media/"

MEDIA_ROOT = os.path.join(PROJECT_DIR, 'uploaded_videos')
# Media URL and Root for Images (New variables for image uploads)
IMAGE_MEDIA_URL = "/media/"
IMAGE_MEDIA_ROOT = os.path.join(PROJECT_DIR, 'image_uploaded')
IM_MEDIA_URL = '/media/'
IM_MEDIA_ROOT = os.path.join(PROJECT_DIR, 'image_uploaded')
TEXT_UPLOAD_FOLDER = os.path.join(PROJECT_DIR, 'text_uploaded')

#for extra logging in production environment
if DEBUG == False:
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
            },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'log.django',
        },
        },
        'loggers': {
            'django': {
                'handlers': ['console','file'],
                'level': os.getenv('DJANGO_LOG_LEVEL', 'DEBUG'),
            },
        },
    }
