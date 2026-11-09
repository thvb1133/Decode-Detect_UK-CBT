package dicegame;

// One person at the keyboard. The sheet stores their column entries.
// This object only keeps the name and the points won from columns.
public class Player {

    private final String name;
    private int score;

    Player(String name) {
        this.name = name;
    }

    String getName() {
        return name;
    }

    int getScore() {
        return score;
    }

    // Points come from the column header, not from the dice total.
    void addScore(int points) {
        score = score + points;
    }
}
