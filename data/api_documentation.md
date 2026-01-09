# TaskFlow Pro - API Technical Documentation

## API Overview

The TaskFlow Pro REST API allows developers to integrate TaskFlow functionality into their applications, automate workflows, and build custom integrations. Our API follows REST principles and returns JSON responses.

**Base URL:** `https://api.taskflowpro.com/v1`

**Current Version:** v1 (released January 2025)

**API Stability:** Production-ready with backwards compatibility guarantee

## Authentication

### API Keys

All API requests require authentication using API keys. Generate keys from your account settings at Settings > Developers > API Keys.

**Authentication Methods:**

**Bearer Token (Recommended):**
```
Authorization: Bearer YOUR_API_KEY
```

**Header Parameter:**
```
X-API-Key: YOUR_API_KEY
```

### Generating API Keys

1. Navigate to Settings > Developers > API Keys
2. Click "Generate New Key"
3. Provide a descriptive name (e.g., "Production Integration")
4. Select appropriate scopes/permissions
5. Copy the key immediately (shown only once)
6. Store securely in environment variables

**Security Best Practices:**
- Never commit API keys to version control
- Rotate keys every 90 days
- Use separate keys for development and production
- Revoke compromised keys immediately
- Use read-only keys when write access isn't needed

### API Key Scopes

Limit API key permissions to minimum required access:

- `tasks:read` - Read task information
- `tasks:write` - Create and update tasks
- `projects:read` - Read project information
- `projects:write` - Create and update projects
- `users:read` - Read user information
- `users:write` - Manage users (admin only)
- `reports:read` - Access reporting data
- `webhooks:manage` - Create and manage webhooks
- `admin:all` - Full administrative access

## Rate Limiting

### Standard Rate Limits

**By Plan:**
- Starter: 100 requests per hour per API key
- Professional: 1,000 requests per hour per API key
- Business: 10,000 requests per hour per API key
- Enterprise: Custom limits (contact sales)

**Rate Limit Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1640995200
```

### Handling Rate Limits

When you exceed rate limits, you'll receive a 429 response:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Retry after 3600 seconds.",
  "retry_after": 3600
}
```

**Best Practices:**
- Implement exponential backoff for retries
- Cache responses when possible
- Use webhooks instead of polling
- Batch requests when supported
- Monitor rate limit headers

### Increasing Limits

Business and Enterprise customers can request higher rate limits:
- Email api-support@taskflowpro.com
- Provide use case and expected request volume
- Approval typically within 2 business days

## Core Endpoints

### Tasks

#### Get All Tasks
```
GET /tasks
```

**Query Parameters:**
- `project_id` (string): Filter by project
- `assignee_id` (string): Filter by assigned user
- `status` (string): Filter by status (open, in_progress, completed)
- `due_date_start` (date): Filter by due date range start
- `due_date_end` (date): Filter by due date range end
- `limit` (integer): Results per page (default: 50, max: 100)
- `offset` (integer): Pagination offset

**Example Request:**
```bash
curl -X GET "https://api.taskflowpro.com/v1/tasks?project_id=proj_abc123&status=open" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Example Response:**
```json
{
  "data": [
    {
      "id": "task_xyz789",
      "title": "Update API documentation",
      "description": "Add examples for new endpoints",
      "status": "in_progress",
      "priority": "high",
      "assignee_id": "user_123",
      "project_id": "proj_abc123",
      "due_date": "2026-01-15T17:00:00Z",
      "created_at": "2026-01-08T10:00:00Z",
      "updated_at": "2026-01-08T14:30:00Z",
      "tags": ["documentation", "api"],
      "time_estimate": 480,
      "time_spent": 120
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

#### Create Task
```
POST /tasks
```

**Request Body:**
```json
{
  "title": "Implement new feature",
  "description": "Build user authentication flow",
  "project_id": "proj_abc123",
  "assignee_id": "user_456",
  "due_date": "2026-01-20T17:00:00Z",
  "priority": "high",
  "status": "open",
  "tags": ["feature", "backend"],
  "time_estimate": 960
}
```

**Response:** 201 Created
```json
{
  "id": "task_new123",
  "title": "Implement new feature",
  "status": "open",
  "created_at": "2026-01-08T15:00:00Z",
  "url": "https://app.taskflowpro.com/tasks/task_new123"
}
```

#### Update Task
```
PATCH /tasks/{task_id}
```

**Request Body:**
```json
{
  "status": "completed",
  "time_spent": 900
}
```

**Response:** 200 OK with updated task object

#### Delete Task
```
DELETE /tasks/{task_id}
```

**Response:** 204 No Content

### Projects

#### List Projects
```
GET /projects
```

**Query Parameters:**
- `status` (string): Filter by status (active, archived, completed)
- `team_id` (string): Filter by team
- `limit` (integer): Results per page
- `offset` (integer): Pagination offset

**Example Response:**
```json
{
  "data": [
    {
      "id": "proj_abc123",
      "name": "Q1 Product Launch",
      "description": "Launch new features for Q1",
      "status": "active",
      "owner_id": "user_789",
      "team_id": "team_xyz",
      "start_date": "2026-01-01T00:00:00Z",
      "due_date": "2026-03-31T23:59:59Z",
      "created_at": "2025-12-15T10:00:00Z",
      "task_count": 45,
      "completed_tasks": 12,
      "progress_percent": 27
    }
  ]
}
```

#### Create Project
```
POST /projects
```

**Request Body:**
```json
{
  "name": "Website Redesign",
  "description": "Complete overhaul of company website",
  "owner_id": "user_123",
  "team_id": "team_design",
  "start_date": "2026-02-01T00:00:00Z",
  "due_date": "2026-05-31T23:59:59Z",
  "template_id": "tmpl_marketing" 
}
```

### Users

#### Get Current User
```
GET /users/me
```

**Response:**
```json
{
  "id": "user_123",
  "email": "john@company.com",
  "name": "John Smith",
  "role": "developer",
  "avatar_url": "https://cdn.taskflowpro.com/avatars/user_123.jpg",
  "timezone": "America/New_York",
  "created_at": "2025-06-15T10:00:00Z"
}
```

#### List Team Users
```
GET /users
```

**Query Parameters:**
- `team_id` (string): Filter by team
- `role` (string): Filter by role
- `status` (string): active or inactive

### Time Tracking

#### Log Time Entry
```
POST /time-entries
```

**Request Body:**
```json
{
  "task_id": "task_xyz789",
  "user_id": "user_123",
  "duration": 7200,
  "description": "Implemented authentication logic",
  "started_at": "2026-01-08T09:00:00Z",
  "billable": true
}
```

**Response:** 201 Created
```json
{
  "id": "time_entry_456",
  "task_id": "task_xyz789",
  "duration": 7200,
  "billable": true,
  "created_at": "2026-01-08T17:00:00Z"
}
```

#### Get Time Entries
```
GET /time-entries
```

**Query Parameters:**
- `task_id` (string): Filter by task
- `user_id` (string): Filter by user
- `project_id` (string): Filter by project
- `start_date` (date): Date range start
- `end_date` (date): Date range end
- `billable` (boolean): Filter billable entries

### Comments

#### Add Comment to Task
```
POST /tasks/{task_id}/comments
```

**Request Body:**
```json
{
  "text": "I've completed the initial implementation. Please review.",
  "mentions": ["user_456", "user_789"]
}
```

**Response:** 201 Created
```json
{
  "id": "comment_abc",
  "task_id": "task_xyz789",
  "user_id": "user_123",
  "text": "I've completed the initial implementation. Please review.",
  "mentions": ["user_456", "user_789"],
  "created_at": "2026-01-08T16:30:00Z"
}
```

## Webhooks

### Creating Webhooks

Webhooks allow you to receive real-time notifications when events occur in TaskFlow Pro.

```
POST /webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhooks/taskflow",
  "events": [
    "task.created",
    "task.updated",
    "task.completed",
    "comment.created"
  ],
  "active": true
}
```

### Webhook Events

Available event types:
- `task.created` - New task created
- `task.updated` - Task updated
- `task.completed` - Task marked complete
- `task.deleted` - Task deleted
- `project.created` - New project created
- `project.updated` - Project updated
- `comment.created` - Comment added
- `user.added` - User added to workspace
- `time_entry.logged` - Time entry logged

### Webhook Payload

Example webhook payload for `task.created`:

```json
{
  "event": "task.created",
  "timestamp": "2026-01-08T15:00:00Z",
  "webhook_id": "webhook_123",
  "data": {
    "task": {
      "id": "task_new123",
      "title": "New task",
      "project_id": "proj_abc123",
      "assignee_id": "user_456",
      "created_by": "user_123",
      "created_at": "2026-01-08T15:00:00Z"
    }
  }
}
```

### Webhook Security

Verify webhook authenticity using the signature header:

```
X-TaskFlow-Signature: sha256=<signature>
```

**Verification Example (Python):**
```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## Error Handling

### Error Response Format

```json
{
  "error": "validation_error",
  "message": "Invalid request parameters",
  "details": [
    {
      "field": "due_date",
      "message": "Due date must be in the future"
    }
  ],
  "request_id": "req_abc123"
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `204 No Content` - Delete successful
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Invalid or missing API key
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource doesn't exist
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Temporary outage

### Common Error Types

- `authentication_error` - Invalid API key
- `authorization_error` - Insufficient permissions
- `validation_error` - Invalid parameters
- `not_found` - Resource not found
- `rate_limit_exceeded` - Too many requests
- `server_error` - Internal error

## Pagination

All list endpoints support cursor-based pagination:

**Request:**
```
GET /tasks?limit=50&offset=0
```

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "total": 500,
    "limit": 50,
    "offset": 0,
    "has_more": true,
    "next_offset": 50
  }
}
```

**Next Page:**
```
GET /tasks?limit=50&offset=50
```

## Filtering and Sorting

### Filtering

Use query parameters to filter results:
```
GET /tasks?status=open&priority=high&assignee_id=user_123
```

### Sorting

Use `sort` parameter with field name and direction:
```
GET /tasks?sort=due_date:asc
GET /tasks?sort=created_at:desc
GET /tasks?sort=priority:desc,due_date:asc
```

## Batch Operations

### Batch Create
```
POST /tasks/batch
```

**Request Body:**
```json
{
  "tasks": [
    {
      "title": "Task 1",
      "project_id": "proj_abc123"
    },
    {
      "title": "Task 2",
      "project_id": "proj_abc123"
    }
  ]
}
```

Maximum 100 items per batch request.

## SDKs and Libraries

Official SDKs available:

**JavaScript/Node.js:**
```bash
npm install @taskflowpro/api-client
```

**Python:**
```bash
pip install taskflowpro
```

**Ruby:**
```bash
gem install taskflowpro
```

**PHP:**
```bash
composer require taskflowpro/api-client
```

**Example Usage (JavaScript):**
```javascript
const TaskFlow = require('@taskflowpro/api-client');
const client = new TaskFlow('YOUR_API_KEY');

// Create a task
const task = await client.tasks.create({
  title: 'New task from API',
  project_id: 'proj_abc123',
  assignee_id: 'user_456'
});

console.log(task.id);
```

## Support

**Documentation:** https://developers.taskflowpro.com

**API Status:** https://status.taskflowpro.com

**Support:**
- Email: api-support@taskflowpro.com
- Developer Slack: slack.taskflowpro.com/developers
- GitHub Issues: github.com/taskflowpro/api-issues

**Response Times:**
- Professional: 24 hours
- Business: 4 hours
- Enterprise: 1 hour

Last Updated: January 2026
API Version: v1.0
