# Initialize the system
cns = CentralNervousSystem(user_id="user_123")

# Example stimulus: User has a busy day and an important meeting
stimulus = {
    "type": "schedule_conflict",
    "data": {
        "current_schedule": "packed",
        "new_meeting": "high_importance", 
        "user_energy_level": "low",
        "historical_preference": "protect_focus_time"
    }
}

# Process through the central nervous system
result = await cns.process_stimulus(
    stimulus=stimulus,
    context={"time_of_day": "morning", "user_stress_level": "medium"}
)

print(result.explanation)
# Output might be:
"""
**Action Taken**: Declined the meeting and scheduled focus time
**Confidence Level**: 85%
**Key Factors**:
- Historical data shows you perform best with protected morning focus time
- Current schedule already exceeds optimal meeting density
- Meeting importance doesn't override focus protection rule
**Decision Type**: Autonomous Execute
"""