import os

# Add the static files directory
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# YouTube API Key
YOUTUBE_API_KEY = 'AIzaSyBtNM-yJuuMO_ZU6gioUP7Ey2iSb6BaFTA'