# Instructions to Run the Code  
## Complete Instructions:  
### Step 1: Install Dependencies
pip install -r requirements.txt

### Step 2 (Optional and One-Time): Create Vector Embeddings and Upsert to Pinecone
python initialize.py

### Step 3: Run the Django Development Server
python manage.py runserver

## If the Above Steps Don't Work, Use a Virtual Environment:
### Step 1: Create and Activate a Virtual Environment
virtualenv venv
source venv/bin/activate

### Step 2: Install Django
pip install django

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4 (Optional and One-Time): Create Vector Embeddings and Upsert to Pinecone
python initialize.py

### Step 5: Run the Django Development Server
python manage.py runserver
