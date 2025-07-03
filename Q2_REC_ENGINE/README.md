# AI-Powered Recommendation Engine

A sophisticated e-commerce recommendation system built with Flask, featuring multiple recommendation algorithms, user authentication, and a modern web interface.

## 🚀 Features

- **Multi-Algorithm Recommendation System**: Content-based, collaborative filtering, and popularity-based recommendations
- **Hybrid Recommendation Engine**: Combines multiple approaches for optimal results
- **User Authentication**: Secure registration and login system
- **Interactive Product Browsing**: View, like, and purchase products
- **Real-time Recommendations**: Dynamic recommendations based on user interactions
- **Modern Web Interface**: Responsive design with Bootstrap
- **Comprehensive Test Suite**: Unit tests for authentication and recommendation functionality

## 🚀 Architecture

### Database Models

The system uses SQLAlchemy with three main models:

- **User**: Stores user credentials and profile information
- **Product**: Comprehensive product catalog with 15+ attributes
- **UserInteraction**: Tracks user behavior (views, likes, purchases)

### Recommendation Engine

The `RecommendationEngine` class implements three core recommendation algorithms:

#### 1. Content-Based Filtering
- Uses TF-IDF vectorization on product text features
- Calculates cosine similarity between products
- Features: product name, category, subcategory, description, manufacturer
- Returns products similar to a given product

#### 2. Collaborative Filtering
- Builds user-item interaction matrix
- Uses weighted interaction scores (view=1, like=3, purchase=5)
- Finds similar users using cosine similarity
- Recommends products liked by similar users

#### 3. Popularity-Based Filtering
- Sorts products by featured status, rating, and sale status
- Provides baseline recommendations for new users
- Ensures popular items are always visible

#### 4. Hybrid Recommendations
- Combines all three approaches
- Removes duplicates and ranks by combined scores
- Provides diverse and relevant recommendations

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Database**: SQLite with SQLAlchemy ORM
- **Machine Learning**: scikit-learn (TF-IDF, cosine similarity)
- **Data Processing**: pandas, numpy
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Authentication**: Werkzeug password hashing
- **Testing**: Python unittest framework

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Q2_REC_ENGINE
   ```

2. **Install dependencies**
   ```bash
   pip install flask flask-sqlalchemy scikit-learn pandas numpy werkzeug
   ```

3. **Initialize the database**
   ```bash
   python app.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open http://localhost:5000 in your browser
   - Register a new account or use existing credentials

## 🧪 Test Suite

The project includes comprehensive unit tests in `test_app.py`:

### Authentication Tests
- User registration
- User login/logout
- Password validation
- Session management

### Recommendation Tests
- Recommendation engine functionality
- User interaction tracking
- Product recommendation generation

### Running Tests
```bash
python test_app.py
```

## 📊 Data Structure

### Sample Data
The system includes a comprehensive `data.json` file with:
- 1000+ products across multiple categories
- Rich product attributes (price, rating, manufacturer, etc.)
- Realistic product descriptions and metadata

### Product Categories
- Home & Garden
- Food & Beverages
- Electronics
- Clothing & Fashion
- Health & Beauty
- Sports & Outdoors
- And more...

## 📄 API Endpoints

### Authentication
- `GET/POST /register` - User registration
- `GET/POST /login` - User authentication
- `GET /logout` - User logout

### Product Management
- `GET /products` - Browse products with filtering
- `GET /product/<id>` - Product details
- `POST /like/<id>` - Like a product
- `POST /purchase/<id>` - Purchase a product

### Recommendations
- `GET /recommendations` - Personalized recommendations
- `GET /` - Homepage with hybrid recommendations

### Debug & Development
- `GET /add_sample_data` - Load sample data
- `GET /debug_interactions` - View user interactions
- `GET /debug_recommendations` - Test recommendation algorithms
- `GET /reset_and_reload` - Reset database and reload data

## 🎯 Recommendation Algorithm Details

### Content-Based Filtering Implementation

```python
def get_content_based_recommendations(self, product_id, n_recommendations=5):
    # TF-IDF vectorization of product features
    text_features = (
        self.products_df['product_name'] + ' ' +
        self.products_df['category'] + ' ' +
        self.products_df['subcategory'] + ' ' +
        self.products_df['description'] + ' ' +
        self.products_df['manufacturer']
    )
    
    # Calculate cosine similarity
    product_similarities = cosine_similarity(
        self.tfidf_matrix[product_idx:product_idx+1], 
        self.tfidf_matrix
    ).flatten()
```

### Collaborative Filtering Implementation

```python
def get_collaborative_recommendations(self, user_id, n_recommendations=5):
    # Build user-item matrix with weighted interactions
    interaction_weights = {'view': 1, 'like': 3, 'purchase': 5}
    
    # Find similar users using cosine similarity
    user_similarities = cosine_similarity(
        self.user_item_matrix.loc[user_id:user_id],
        self.user_item_matrix
    ).flatten()
```

### Hybrid Recommendation Strategy

```python
def get_hybrid_recommendations(self, user_id, product_id=None, n_recommendations=10):
    # Combine multiple recommendation approaches
    content_recs = self.get_content_based_recommendations(product_id, n_recommendations//3)
    collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations//3)
    popular_recs = self.get_popularity_recommendations(n_recommendations//3)
    
    # Remove duplicates and rank by combined scores
    return unique_recommendations[:n_recommendations]
```

## 🔍 Cursor-Assisted Development Process

This project was developed using Cursor IDE with AI assistance, demonstrating modern development practices:

### 1. **Iterative Development**
- Started with basic Flask application structure
- Gradually added recommendation algorithms
- Implemented user authentication and interaction tracking
- Built comprehensive test suite

### 2. **AI-Powered Code Generation**
- Used Cursor's AI to generate database models
- Assisted with recommendation algorithm implementation
- Generated test cases and documentation
- Optimized code structure and performance

### 3. **Code Quality Assurance**
- Implemented comprehensive error handling
- Added input validation and security measures
- Created modular, maintainable code structure
- Ensured proper separation of concerns

### 4. **Testing Strategy**
- Unit tests for core functionality
- Integration tests for recommendation algorithms
- Authentication flow testing
- Database operation validation

### 5. **Documentation**
- Comprehensive code comments
- API endpoint documentation
- Algorithm explanation and implementation details
- Setup and deployment instructions

## 🚀 Performance Optimizations

- **Efficient Data Loading**: Products loaded into memory for faster recommendations
- **Caching Strategy**: TF-IDF matrices computed once and reused
- **Database Indexing**: Optimized queries with proper foreign key relationships
- **Lazy Loading**: User-item matrices built on-demand

## 🔒 Security Features

- **Password Hashing**: Secure password storage using Werkzeug
- **Session Management**: Flask session-based authentication
- **Input Validation**: Form validation and sanitization
- **SQL Injection Prevention**: SQLAlchemy ORM protection

## 📈 Future Enhancements

- **Real-time Recommendations**: WebSocket-based live updates
- **Advanced ML Models**: Deep learning recommendation algorithms
- **A/B Testing**: Recommendation algorithm comparison
- **Analytics Dashboard**: User behavior insights
- **Mobile API**: RESTful API for mobile applications
- **Scalability**: Redis caching and database optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤖 Author

Developed with assistance from Cursor IDE's AI capabilities, demonstrating the power of AI-assisted development in creating sophisticated recommendation systems.

---

*This README provides a comprehensive overview of the recommendation engine implementation, test suite, and development process. The project showcases modern web development practices combined with machine learning algorithms for personalized product recommendations.* 