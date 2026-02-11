
---
---

## API Endpoints

### Health Check
GET /health

Response:
{ "status": "ok" }

---

### Get Recommendations
GET /recommendations/{user_id}

Returns top 10 recommended movies.

If the user does not exist, cold-start recommendations are returned.


