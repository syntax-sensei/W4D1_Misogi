import unittest
from app import app, db, User, Product, UserInteraction

class AuthTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        with app.app_context():
            db.create_all()
            # Add a test user
            user = User(username='testuser', email='test@example.com', password_hash='pbkdf2:sha256:150000$test$test')
            db.session.add(user)
            db.session.commit()

    def tearDown(self):
        with app.app_context():
            db.drop_all()

    def test_register_login_logout(self):
        # Register
        rv = self.app.post('/register', data=dict(
            username='newuser',
            email='new@example.com',
            password='password'
        ), follow_redirects=True)
        self.assertIn(b'Registration successful', rv.data)
        # Login
        rv = self.app.post('/login', data=dict(
            username='newuser',
            password='password'
        ), follow_redirects=True)
        self.assertIn(b'Login successful', rv.data)
        # Logout
        rv = self.app.get('/logout', follow_redirects=True)
        self.assertIn(b'Logged out successfully', rv.data)

class RecommendationTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        with app.app_context():
            db.create_all()
            # Add test data
            user = User(username='testuser', email='test@example.com', password_hash='pbkdf2:sha256:150000$test$test')
            db.session.add(user)
            db.session.commit()
            product = Product(product_id=1, product_name='Test Product', category='Test', subcategory='Test', price=10.0, quantity_in_stock=10, manufacturer='Test', description='Test', weight=1.0, dimensions='1x1', release_date='2020-01-01', rating=5.0, is_featured=True, is_on_sale=False, sale_price=10.0, image_url='')
            db.session.add(product)
            db.session.commit()
            interaction = UserInteraction(user_id=user.id, product_id=product.id, interaction_type='like')
            db.session.add(interaction)
            db.session.commit()

    def tearDown(self):
        with app.app_context():
            db.drop_all()

    def test_recommendations(self):
        with self.app.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'testuser'
        rv = self.app.get('/recommendations')
        self.assertIn(b'AI-Powered Recommendations', rv.data)

if __name__ == '__main__':
    unittest.main() 