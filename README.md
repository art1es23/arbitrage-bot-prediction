- AI
  source venv/bin/activate
  pip install Flask scikit-learn pandas tensorflow
  python3 ./services/model.py
- Backend
  mongod --config /opt/homebrew/etc/mongod.conf --fork
  npm run server
- Frontend
  npm run dev
