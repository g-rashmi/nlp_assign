

# 🧠 NLP Assign (chat with pdf)

---

##  How to Run the Application

###  Backend Setup (FastAPI)

1. Navigate to the `backend` directory:

   ```bash
   cd backend
   ```

2. Create a `.env` file and add your Google API credentials:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   >  You can generate the API key from [Google AI Studio](https://makersuite.google.com/app).

3. Set up and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate     # For Windows
 
4. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the backend server:

   ```bash
   uvicorn main:app --reload
   ```

---

### Frontend Setup (React)

1. Navigate to the `frontend` directory:

   ```bash
   cd frontend
   ```

2. Install frontend dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```
