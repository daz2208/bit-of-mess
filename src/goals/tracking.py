"""Goal tracking system for long-term objectives."""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid


@dataclass
class Milestone:
    """A milestone toward a goal."""
    id: str
    description: str
    target_date: datetime | None
    completed: bool
    completed_at: datetime | None
    progress: float  # 0-1


@dataclass
class Goal:
    """A tracked goal."""
    id: str
    user_id: str
    title: str
    description: str
    category: str
    created_at: datetime
    target_date: datetime | None
    milestones: list[Milestone]
    progress: float  # 0-1
    status: str  # "active", "completed", "paused", "abandoned"
    priority: int  # 1-10
    related_preferences: list[str]
    check_ins: list[dict]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "progress": self.progress,
            "status": self.status,
            "priority": self.priority,
            "milestones_completed": sum(1 for m in self.milestones if m.completed),
            "milestones_total": len(self.milestones)
        }


class GoalTracker:
    """
    Track long-term user goals and progress.

    Features:
    - Goal decomposition into milestones
    - Progress tracking
    - Periodic check-ins
    - Goal-preference alignment
    """

    def __init__(self):
        self.goals: dict[str, dict[str, Goal]] = {}  # user_id -> goal_id -> Goal

    def create_goal(
        self,
        user_id: str,
        title: str,
        description: str = "",
        category: str = "general",
        target_date: datetime | None = None,
        milestones: list[str] = None,
        priority: int = 5
    ) -> Goal:
        """Create a new goal."""
        if user_id not in self.goals:
            self.goals[user_id] = {}

        # Create milestones
        milestone_objs = []
        if milestones:
            for i, desc in enumerate(milestones):
                milestone_objs.append(Milestone(
                    id=str(uuid.uuid4()),
                    description=desc,
                    target_date=None,
                    completed=False,
                    completed_at=None,
                    progress=0.0
                ))

        goal = Goal(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            description=description,
            category=category,
            created_at=datetime.utcnow(),
            target_date=target_date,
            milestones=milestone_objs,
            progress=0.0,
            status="active",
            priority=priority,
            related_preferences=[],
            check_ins=[]
        )

        self.goals[user_id][goal.id] = goal
        return goal

    def update_progress(
        self,
        user_id: str,
        goal_id: str,
        progress: float | None = None,
        milestone_id: str | None = None,
        milestone_progress: float | None = None
    ):
        """Update goal or milestone progress."""
        if user_id not in self.goals or goal_id not in self.goals[user_id]:
            return

        goal = self.goals[user_id][goal_id]

        if milestone_id:
            # Update specific milestone
            for milestone in goal.milestones:
                if milestone.id == milestone_id:
                    if milestone_progress is not None:
                        milestone.progress = min(1.0, milestone_progress)
                    if milestone_progress >= 1.0:
                        milestone.completed = True
                        milestone.completed_at = datetime.utcnow()
                    break

            # Recalculate goal progress from milestones
            if goal.milestones:
                goal.progress = sum(m.progress for m in goal.milestones) / len(goal.milestones)

        elif progress is not None:
            goal.progress = min(1.0, progress)

        # Check if goal completed
        if goal.progress >= 1.0:
            goal.status = "completed"

        # Record check-in
        goal.check_ins.append({
            "timestamp": datetime.utcnow().isoformat(),
            "progress": goal.progress,
            "note": f"Updated to {goal.progress:.0%}"
        })

    def complete_milestone(self, user_id: str, goal_id: str, milestone_id: str):
        """Mark a milestone as completed."""
        self.update_progress(user_id, goal_id, milestone_id=milestone_id, milestone_progress=1.0)

    def get_active_goals(self, user_id: str) -> list[Goal]:
        """Get all active goals for a user."""
        if user_id not in self.goals:
            return []

        return [g for g in self.goals[user_id].values() if g.status == "active"]

    def get_goal(self, user_id: str, goal_id: str) -> Goal | None:
        """Get a specific goal."""
        if user_id not in self.goals:
            return None
        return self.goals[user_id].get(goal_id)

    def get_upcoming_milestones(
        self,
        user_id: str,
        days_ahead: int = 7
    ) -> list[tuple[Goal, Milestone]]:
        """Get milestones due in the next N days."""
        if user_id not in self.goals:
            return []

        upcoming = []
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)

        for goal in self.goals[user_id].values():
            if goal.status != "active":
                continue

            for milestone in goal.milestones:
                if not milestone.completed and milestone.target_date:
                    if milestone.target_date <= cutoff:
                        upcoming.append((goal, milestone))

        # Sort by date
        upcoming.sort(key=lambda x: x[1].target_date or datetime.max)
        return upcoming

    def get_stalled_goals(
        self,
        user_id: str,
        stall_days: int = 7
    ) -> list[Goal]:
        """Get goals with no progress in N days."""
        if user_id not in self.goals:
            return []

        stalled = []
        cutoff = datetime.utcnow() - timedelta(days=stall_days)

        for goal in self.goals[user_id].values():
            if goal.status != "active":
                continue

            # Check last check-in
            if goal.check_ins:
                last_checkin = datetime.fromisoformat(goal.check_ins[-1]["timestamp"])
                if last_checkin < cutoff:
                    stalled.append(goal)
            elif goal.created_at < cutoff:
                stalled.append(goal)

        return stalled

    def suggest_next_actions(self, user_id: str) -> list[dict]:
        """Suggest next actions based on goals."""
        suggestions = []

        active_goals = self.get_active_goals(user_id)

        for goal in active_goals:
            # Find next incomplete milestone
            next_milestone = None
            for milestone in goal.milestones:
                if not milestone.completed:
                    next_milestone = milestone
                    break

            if next_milestone:
                suggestions.append({
                    "goal": goal.title,
                    "action": f"Work on: {next_milestone.description}",
                    "progress": next_milestone.progress,
                    "priority": goal.priority
                })
            else:
                # No milestones, suggest working on goal directly
                suggestions.append({
                    "goal": goal.title,
                    "action": f"Continue progress on: {goal.title}",
                    "progress": goal.progress,
                    "priority": goal.priority
                })

        # Sort by priority
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        return suggestions[:5]

    def check_in(self, user_id: str, goal_id: str, note: str):
        """Record a check-in for a goal."""
        if user_id not in self.goals or goal_id not in self.goals[user_id]:
            return

        goal = self.goals[user_id][goal_id]
        goal.check_ins.append({
            "timestamp": datetime.utcnow().isoformat(),
            "progress": goal.progress,
            "note": note
        })

    def get_progress_summary(self, user_id: str) -> dict:
        """Get summary of all goal progress."""
        if user_id not in self.goals:
            return {"total": 0, "active": 0, "completed": 0, "avg_progress": 0}

        goals = list(self.goals[user_id].values())

        active = [g for g in goals if g.status == "active"]
        completed = [g for g in goals if g.status == "completed"]

        avg_progress = 0.0
        if active:
            avg_progress = sum(g.progress for g in active) / len(active)

        return {
            "total": len(goals),
            "active": len(active),
            "completed": len(completed),
            "avg_progress": avg_progress,
            "by_category": self._group_by_category(goals)
        }

    def _group_by_category(self, goals: list[Goal]) -> dict:
        """Group goals by category."""
        categories = {}
        for goal in goals:
            if goal.category not in categories:
                categories[goal.category] = {"count": 0, "avg_progress": 0}
            categories[goal.category]["count"] += 1
            categories[goal.category]["avg_progress"] += goal.progress

        # Calculate averages
        for cat in categories:
            if categories[cat]["count"] > 0:
                categories[cat]["avg_progress"] /= categories[cat]["count"]

        return categories
