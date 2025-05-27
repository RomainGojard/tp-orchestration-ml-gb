# Dockerfile pour le frontend HTML
FROM nginx:alpine

# Supprime le fichier par défaut d'accueil de nginx
RUN rm -rf /usr/share/nginx/html/*

# Copie ton HTML dans le répertoire par défaut de nginx
COPY index.html /usr/share/nginx/html

# Expose le port 80 (nginx)
EXPOSE 80
