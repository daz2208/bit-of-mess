# Example learning scenario
async def example_learning_scenario():
    learner = MultiModalLearner()
    
    # User explicitly corrects the agent
    explicit_feedback = FeedbackEvent(
        type="direct_correction",
        data={
            "wrong_behavior": "scheduled meeting during focus time",
            "correct_behavior": "protect 9-11 AM as focus time"
        },
        context={"time_of_day": "morning", "day_type": "workday"},
        emotional_tone="frustrated"
    )
    
    # User implicitly rejects suggestions
    implicit_feedback = FeedbackEvent(
        type="suggestion_ignored", 
        data={
            "suggestion_type": "meeting_scheduling",
            "content": "Schedule brainstorming at 10 AM",
            "user_action": "ignored"
        }
    )
    
    # Process both feedback types
    updates = []
    for event in [explicit_feedback, implicit_feedback]:
        update = await learner.process_feedback_event(event)
        if update:
            updates.append(update)
    
    # Integrate learning
    integrator = LearningIntegrator()
    results = await integrator.integrate_learning_updates(updates)
    
    # Apply to knowledge bases
    knowledge_updater = ContinuousKnowledgeUpdater()
    await knowledge_updater.update_knowledge(results)
    
    print("Learning completed. Agent now understands:")
    print("- Morning focus time protection (explicit)")
    print("- Avoid scheduling meetings during focus blocks (implicit)")

# Run the learning scenario
await example_learning_scenario()