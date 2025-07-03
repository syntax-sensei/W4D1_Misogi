from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'NcVJK7TCnBlnvHby5EriSgNDRx4vvyDW'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recommendation_engine.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    interactions = db.relationship('UserInteraction', backref='user', lazy=True)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, unique=True, nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    subcategory = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity_in_stock = db.Column(db.Integer, nullable=False)
    manufacturer = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    dimensions = db.Column(db.String(200), nullable=False)
    release_date = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    is_featured = db.Column(db.Boolean, default=False)
    is_on_sale = db.Column(db.Boolean, default=False)
    sale_price = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(500), nullable=False)
    
    # Relationships
    interactions = db.relationship('UserInteraction', backref='product', lazy=True)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)  # 'view', 'like', 'purchase'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Recommendation Engine Class
class RecommendationEngine:
    def __init__(self):
        self.products_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.user_item_matrix = None
        
    def load_products(self):
        """Load products from database into DataFrame"""
        products = Product.query.all()
        self.products_df = pd.DataFrame([{
            'id': p.id,
            'product_id': p.product_id,
            'product_name': p.product_name,
            'category': p.category,
            'subcategory': p.subcategory,
            'price': p.price,
            'rating': p.rating,
            'is_featured': p.is_featured,
            'is_on_sale': p.is_on_sale,
            'description': p.description,
            'manufacturer': p.manufacturer
        } for p in products])
        
        # Create TF-IDF matrix for content-based filtering
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Combine text features for TF-IDF
        text_features = (
            self.products_df['product_name'] + ' ' +
            self.products_df['category'] + ' ' +
            self.products_df['subcategory'] + ' ' +
            self.products_df['description'] + ' ' +
            self.products_df['manufacturer']
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
    def build_user_item_matrix(self):
        """Build user-item interaction matrix for collaborative filtering"""
        interactions = UserInteraction.query.all()
        
        if not interactions:
            self.user_item_matrix = None
            return
            
        interaction_data = []
        for interaction in interactions:
            interaction_data.append({
                'user_id': interaction.user_id,
                'product_id': interaction.product_id,
                'interaction_type': interaction.interaction_type
            })
        
        interactions_df = pd.DataFrame(interaction_data)
        
        # Create weighted interaction scores
        interaction_weights = {'view': 1, 'like': 3, 'purchase': 5}
        interactions_df['weight'] = interactions_df['interaction_type'].map(interaction_weights)
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='weight',
            fill_value=0
        )
    
    def get_content_based_recommendations(self, product_id, n_recommendations=5):
        """Get content-based recommendations based on product similarity"""
        # Always load fresh data
        self.load_products()
        
        if self.products_df is None or len(self.products_df) == 0:
            return []
        
        # Find product index
        product_idx = self.products_df[self.products_df['product_id'] == product_id].index
        if len(product_idx) == 0:
            return []
        
        product_idx = product_idx[0]
        
        # Calculate cosine similarity
        product_similarities = cosine_similarity(
            self.tfidf_matrix[product_idx:product_idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar products (excluding the product itself)
        similar_indices = product_similarities.argsort()[-n_recommendations-1:-1][::-1]
        
        recommendations = []
        for idx in similar_indices:
            product = self.products_df.iloc[idx]
            recommendations.append({
                'product_id': int(product['product_id']),
                'product_name': product['product_name'],
                'category': product['category'],
                'price': float(product['price']),
                'rating': float(product['rating']),
                'similarity_score': float(product_similarities[idx])
            })
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Get collaborative filtering recommendations"""
        # Always build fresh user-item matrix
        self.build_user_item_matrix()
        
        if self.user_item_matrix is None or len(self.user_item_matrix) == 0:
            return []
        
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's interaction vector
        user_vector = self.user_item_matrix.loc[user_id]
        
        # Find similar users
        user_similarities = cosine_similarity(
            self.user_item_matrix.loc[user_id:user_id],
            self.user_item_matrix
        ).flatten()
        
        # Get top similar users
        similar_users = user_similarities.argsort()[-6:-1][::-1]  # Top 5 similar users
        
        # Get products liked by similar users
        similar_user_ids = self.user_item_matrix.index[similar_users]
        recommendations = defaultdict(float)
        
        for similar_user_id in similar_user_ids:
            similar_user_vector = self.user_item_matrix.loc[similar_user_id]
            similarity_score = user_similarities[similar_users[list(similar_user_ids).index(similar_user_id)]]
            
            for product_id, rating in similar_user_vector.items():
                if rating > 0 and user_vector[product_id] == 0:  # Product not interacted by current user
                    recommendations[product_id] += rating * similarity_score
        
        # Sort and get top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for product_id, score in sorted_recommendations[:n_recommendations]:
            product = Product.query.filter_by(product_id=product_id).first()
            if product:
                result.append({
                    'product_id': product.product_id,
                    'product_name': product.product_name,
                    'category': product.category,
                    'price': product.price,
                    'rating': product.rating,
                    'collaborative_score': float(score)
                })
        
        return result
    
    def get_popularity_recommendations(self, n_recommendations=5):
        """Get popularity-based recommendations"""
        # Always load fresh data
        self.load_products()
        
        if self.products_df is None or len(self.products_df) == 0:
            return []
        
        # Sort by rating and featured status
        popular_products = self.products_df.sort_values(
            by=['is_featured', 'rating', 'is_on_sale'], 
            ascending=[False, False, False]
        ).head(n_recommendations)
        
        recommendations = []
        for _, product in popular_products.iterrows():
            recommendations.append({
                'product_id': int(product['product_id']),
                'product_name': product['product_name'],
                'category': product['category'],
                'price': float(product['price']),
                'rating': float(product['rating']),
                'is_featured': bool(product['is_featured']),
                'is_on_sale': bool(product['is_on_sale'])
            })
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id, product_id=None, n_recommendations=10):
        """Get hybrid recommendations combining multiple approaches"""
        recommendations = []
        
        # Get content-based recommendations if product_id provided
        if product_id:
            content_recs = self.get_content_based_recommendations(product_id, n_recommendations//3)
            recommendations.extend(content_recs)
        
        # Get collaborative recommendations
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations//3)
        recommendations.extend(collab_recs)
        
        # Get popularity recommendations
        popular_recs = self.get_popularity_recommendations(n_recommendations//3)
        recommendations.extend(popular_recs)
        
        # Remove duplicates and sort by combined score
        seen_products = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec['product_id'] not in seen_products:
                seen_products.add(rec['product_id'])
                unique_recommendations.append(rec)
        
        return unique_recommendations[:n_recommendations]

# Initialize recommendation engine
rec_engine = RecommendationEngine()

# Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get recommendations for logged-in user
    rec_engine.load_products()
    rec_engine.build_user_item_matrix()
    
    recommendations = rec_engine.get_hybrid_recommendations(session['user_id'])
    featured_products = rec_engine.get_popularity_recommendations(8)
    
    return render_template('index.html', 
                         recommendations=recommendations, 
                         featured_products=featured_products)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!')
    return redirect(url_for('login'))

@app.route('/products')
def products():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    category = request.args.get('category', '')
    search = request.args.get('search', '')
    
    query = Product.query
    
    if category:
        query = query.filter_by(category=category)
    
    if search:
        query = query.filter(Product.product_name.contains(search))
    
    products = query.paginate(page=request.args.get('page', 1, type=int), per_page=12)
    
    # Get unique categories for filter
    categories = db.session.query(Product.category).distinct().all()
    categories = [cat[0] for cat in categories]
    
    return render_template('products.html', products=products, categories=categories)

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        flash('Product not found')
        return redirect(url_for('products'))
    
    # Record view interaction
    interaction = UserInteraction(
        user_id=session['user_id'],
        product_id=product.id,
        interaction_type='view'
    )
    db.session.add(interaction)
    db.session.commit()
    
    # Check if user has already liked this product
    existing_like = UserInteraction.query.filter_by(
        user_id=session['user_id'],
        product_id=product.id,
        interaction_type='like'
    ).first()
    
    is_liked = existing_like is not None
    
    # Get similar products
    rec_engine.load_products()
    similar_products = rec_engine.get_content_based_recommendations(product_id, 4)
    
    return render_template('product_detail.html', 
                         product=product, 
                         similar_products=similar_products,
                         is_liked=is_liked)  # Pass the like state

@app.route('/like/<int:product_id>', methods=['POST'])
def like_product(product_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    # Check if already liked
    existing_interaction = UserInteraction.query.filter_by(
        user_id=session['user_id'],
        product_id=product.id,
        interaction_type='like'
    ).first()
    
    if existing_interaction:
        # Unlike
        db.session.delete(existing_interaction)
        db.session.commit()
        return jsonify({'liked': False})
    else:
        # Like
        interaction = UserInteraction(
            user_id=session['user_id'],
            product_id=product.id,
            interaction_type='like'
        )
        db.session.add(interaction)
        db.session.commit()
        return jsonify({'liked': True})

@app.route('/purchase/<int:product_id>', methods=['POST'])
def purchase_product(product_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    # Record purchase interaction
    interaction = UserInteraction(
        user_id=session['user_id'],
        product_id=product.id,
        interaction_type='purchase'
    )
    db.session.add(interaction)
    db.session.commit()
    
    flash('Purchase recorded successfully!')
    return jsonify({'success': True})

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get different types of recommendations (methods now load fresh data)
    content_recs = rec_engine.get_content_based_recommendations(1, 5)  # Example product
    collab_recs = rec_engine.get_collaborative_recommendations(session['user_id'], 5)
    popular_recs = rec_engine.get_popularity_recommendations(5)
    hybrid_recs = rec_engine.get_hybrid_recommendations(session['user_id'], n_recommendations=10)
    
    return render_template('recommendations.html',
                         content_recs=content_recs,
                         collab_recs=collab_recs,
                         popular_recs=popular_recs,
                         hybrid_recs=hybrid_recs)

@app.route('/add_sample_data')
def add_sample_data():
    """Add sample user interactions for testing recommendations"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get some products
    products = Product.query.limit(20).all()
    
    # Add sample interactions for current user
    interaction_types = ['view', 'like', 'purchase']
    for i, product in enumerate(products):
        interaction = UserInteraction(
            user_id=session['user_id'],
            product_id=product.id,
            interaction_type=interaction_types[i % 3]  # Cycle through interaction types
        )
        db.session.add(interaction)
    
    db.session.commit()
    flash('Sample interactions added! Recommendations should now change.')
    return redirect(url_for('recommendations'))

@app.route('/check_like/<int:product_id>')
def check_like_status(product_id):
    """Check if user has liked a product"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    existing_like = UserInteraction.query.filter_by(
        user_id=session['user_id'],
        product_id=product.id,
        interaction_type='like'
    ).first()
    
    return jsonify({'liked': existing_like is not None})

@app.route('/debug_interactions')
def debug_interactions():
    """Debug route to check if interactions are being stored"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get all interactions for current user
    interactions = UserInteraction.query.filter_by(user_id=session['user_id']).all()
    
    debug_info = {
        'user_id': session['user_id'],
        'total_interactions': len(interactions),
        'interactions': []
    }
    
    for interaction in interactions:
        product = Product.query.get(interaction.product_id)
        debug_info['interactions'].append({
            'product_name': product.product_name if product else 'Unknown',
            'interaction_type': interaction.interaction_type,
            'timestamp': interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(debug_info)

@app.route('/debug_recommendations')
def debug_recommendations():
    """Debug route to check recommendation engine data"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    rec_engine.load_products()
    rec_engine.build_user_item_matrix()
    
    debug_info = {
        'total_products': len(rec_engine.products_df) if rec_engine.products_df is not None else 0,
        'user_item_matrix_shape': rec_engine.user_item_matrix.shape if rec_engine.user_item_matrix is not None else 'None',
        'user_in_matrix': session['user_id'] in rec_engine.user_item_matrix.index if rec_engine.user_item_matrix is not None else False,
        'user_interactions': []
    }
    
    if rec_engine.user_item_matrix is not None and session['user_id'] in rec_engine.user_item_matrix.index:
        user_vector = rec_engine.user_item_matrix.loc[session['user_id']]
        user_interactions = user_vector[user_vector > 0]
        debug_info['user_interactions'] = [
            {'product_id': int(pid), 'weight': float(weight)} 
            for pid, weight in user_interactions.items()
        ]
    
    return jsonify(debug_info)

@app.route('/reset_and_reload')
def reset_and_reload():
    """Reset database and reload everything"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Clear all interactions for current user
    UserInteraction.query.filter_by(user_id=session['user_id']).delete()
    db.session.commit()
    
    # Add fresh sample interactions
    products = Product.query.limit(30).all()
    interaction_types = ['view', 'like', 'purchase']
    
    for i, product in enumerate(products):
        interaction = UserInteraction(
            user_id=session['user_id'],
            product_id=product.id,
            interaction_type=interaction_types[i % 3]
        )
        db.session.add(interaction)
    
    db.session.commit()
    
    flash('Database reset and fresh interactions added!')
    return redirect(url_for('debug_interactions'))

# Initialize database and load data
def init_db():
    with app.app_context():
        db.create_all()
        
        # Load products from JSON if database is empty
        if Product.query.count() == 0:
            with open('data.json', 'r') as f:
                products_data = json.load(f)
            
            for product_data in products_data:
                product = Product(
                    product_id=product_data['product_id'],
                    product_name=product_data['product_name'],
                    category=product_data['category'],
                    subcategory=product_data['subcategory'],
                    price=product_data['price'],
                    quantity_in_stock=product_data['quantity_in_stock'],
                    manufacturer=product_data['manufacturer'],
                    description=product_data['description'],
                    weight=product_data['weight'],
                    dimensions=product_data['dimensions'],
                    release_date=product_data['release_date'],
                    rating=product_data['rating'],
                    is_featured=product_data['is_featured'],
                    is_on_sale=product_data['is_on_sale'],
                    sale_price=product_data['sale_price'],
                    image_url=product_data['image_url']
                )
                db.session.add(product)
            
            db.session.commit()
            print(f"Loaded {len(products_data)} products into database")

if __name__ == '__main__':
    init_db()
    app.run(debug=True) 