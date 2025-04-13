# Sports Betting Prediction System

A Flask-based web application that displays sports betting predictions based on historical data analysis. The system shows upcoming matches, predictions, and their outcomes in a clean, modern interface.

## Features

- Modern, responsive UI using Bootstrap 5
- Real-time display of upcoming matches and predictions
- Confidence meter for each prediction
- Historical performance tracking
- AWS Elastic Beanstalk deployment ready
- Integration with AWS Athena for real-time data

## Running the Application

### Local Development

To run the application directly on your machine:

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application at http://localhost:8080

### Docker

#### Option 1: Using Docker directly

1. Build and run with the provided script:
   ```
   ./run_docker.sh
   ```

2. Or manually:
   ```
   docker build -t winner-app .
   docker run -p 8080:8080 winner-app
   ```

3. Access the application at http://localhost:8080

#### Option 2: Using Docker Compose (recommended for development)

1. Run with the provided script:
   ```
   ./run_dev.sh
   ```

2. Or manually:
   ```
   docker-compose up --build
   ```

3. Access the application at http://localhost:8080

### Cloud Deployment

To deploy the application to the cloud:

#### AWS Elastic Beanstalk (with Docker)

##### Option 1: Using the AWS Console (Recommended)

1. Run the deployment script to build, push to ECR, and create the deployment package:
   ```
   ./deploy_to_eb.sh
   ```

2. Follow the instructions printed by the script to upload and deploy via the AWS Console.

##### Option 2: Using the EB CLI (If you have AWS CLI configured)

If you have the EB CLI working correctly, you can use:

1. Initialize an EB Docker environment:
   ```
   eb init -p docker
   ```

2. Create and deploy the environment:
   ```
   eb create winner-app-env
   ```

3. For subsequent deployments:
   ```
   eb deploy
   ```

#### Other Cloud Providers

The Docker container can be deployed to any cloud provider that supports Docker containers, such as:

- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- Heroku (with the container stack)

## Project Structure

```
.
├── app.py                  # Main Flask application
├── db.py                   # Database connection and queries
├── create_sample_parquet.py # Script to generate sample data
├── requirements.txt        # Python dependencies
├── .ebextensions/         # AWS Elastic Beanstalk configuration
├── static/                # Static files (CSS, JS)
│   └── css/
│       └── style.css      # Custom styles
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   └── about.html        # About page
└── README.md             # Project documentation
```

## Database Schema

The application connects to an AWS Athena database with the following schema:

- Database: `winner-db`
- Table: `api_odds`
- Columns:
  - `id`: Unique identifier
  - `home_team`: Home team name
  - `away_team`: Away team name
  - `league`: League name
  - `match_time`: Scheduled match time
  - `prediction`: Predicted outcome
  - `confidence`: Confidence score (0-1)
  - `status`: Match status (pending, won, lost)
  - `home_odds`: Home team odds
  - `draw_odds`: Draw odds
  - `away_odds`: Away team odds
  - `run_time`: Time when the prediction was made

## Future Enhancements

- User authentication system
- Advanced filtering and sorting options
- Historical performance graphs
- API endpoints for data access
- Real-time updates using WebSockets

## License

MIT License