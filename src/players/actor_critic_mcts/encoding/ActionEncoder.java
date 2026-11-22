package players.actor_critic_mcts.encoding;

import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.cityactions.Build;
import core.actions.cityactions.CityAction;
import core.actions.cityactions.LevelUp;
import core.actions.cityactions.ResourceGathering;
import core.actions.cityactions.Spawn;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.DeclareWar;
import core.actions.tribeactions.ResearchTech;
import core.actions.tribeactions.SendStars;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Convert;
import core.actions.unitactions.Move;
import core.actions.unitactions.UnitAction;
import core.actors.Actor;
import core.actors.Tribe;
import core.game.GameState;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Tribes has a variable, state-dependent action list, so the actor can't just output "action #17".
 * Pick one scheme and implement the matching method(s):
 *
 *  (A) Fixed index space: map every action to an index in [0, actionSpaceSize()) built from
 *      (Types.ACTION type, source tile / actor, target tile, extra id like unit type or tech).
 *      The policy head outputs actionSpaceSize() logits; mask the illegal ones.
 *
 *  (B) Action features (implemented): every legal action gets a fixed-length, sparse feature vector;
 *      the network scores each (state graph, action features) pair -> one logit per legal action.
 *
 * Scheme B layout (see featureNames() for every column):
 *
 *   action_type one-hot (all Types.ACTION values)          every action
 *   has_src, src_node, has_tgt, tgt_node                   node ids into StateEncoder's graph (x * size + y),
 *                                                          stored as floats; -1 when absent. In Python:
 *                                                          h[src_node.long()] gathers the tile embedding.
 *   dx, dy                                                 (target - source) / (boardSize - 1), if both exist
 *   star_cost                                              stars the action costs the acting tribe, / 20
 *   building one-hot                                       BUILD
 *   resource one-hot                                       RESOURCE_GATHERING
 *   unit_type one-hot                                      SPAWN
 *   level_up one-hot (Types.CITY_LEVEL_UP)                 LEVEL_UP
 *   tech one-hot + tech_tier                               RESEARCH_TECH
 *   target_tribe (relative slot, same as StateEncoder)     DECLARE_WAR, SEND_STARS
 *   stars_sent                                             SEND_STARS
 *   capture_city, capture_village                          CAPTURE
 *
 * Blocks that don't apply to an action are left at 0; the action_type one-hot says which blocks are live.
 *
 * Tiles used as source / target:
 *   city actions      : city tile -> targetPos (SPAWN / LEVEL_UP: the city tile itself)
 *   MOVE              : unit tile -> destination
 *   ATTACK / CONVERT  : unit tile -> target unit's tile
 *   other unit actions: unit tile -> unit tile (CAPTURE, EXAMINE, RECOVER, HEAL_OTHERS, DISBAND, UPGRADE_*...)
 *   BUILD_ROAD        : no source -> road tile
 *   RESEARCH_TECH, END_TURN, DECLARE_WAR, SEND_STARS: no tiles
 */
public class ActionEncoder {

    private static final float STAR_COST_SCALE = 20f;
    private static final float STARS_SENT_SCALE = 50f;
    private static final float MAX_TECH_TIER = 3f;

    private static final Types.ACTION[] ACTIONS = Types.ACTION.values();
    private static final Types.BUILDING[] BUILDINGS = Types.BUILDING.values();
    private static final Types.RESOURCE[] RESOURCES = Types.RESOURCE.values();
    private static final Types.UNIT[] UNITS = Types.UNIT.values();
    private static final Types.CITY_LEVEL_UP[] LEVEL_UPS = Types.CITY_LEVEL_UP.values();
    private static final Types.TECHNOLOGY[] TECHS = Types.TECHNOLOGY.values();

    private final int maxPlayers;
    private final ArrayList<String> names = new ArrayList<>();

    private final int A_TYPE, A_PTR, A_DELTA, A_COST, A_BUILDING, A_RESOURCE, A_UNIT, A_LEVEL_UP,
            A_TECH, A_TECH_TIER, A_TARGET_TRIBE, A_STARS_SENT, A_CAPTURE;

    public ActionEncoder() {
        this(StateEncoder.DEFAULT_MAX_PLAYERS);
    }

    /** maxPlayers must match the StateEncoder used alongside this encoder. */
    public ActionEncoder(int maxPlayers) {
        this.maxPlayers = maxPlayers;
        String[] slots = new String[maxPlayers];
        slots[0] = "self";
        for (int k = 1; k < maxPlayers; k++) slots[k] = "opp" + k;

        A_TYPE = add("action_type", enumNames(ACTIONS));
        A_PTR = add("ptr", "has_src", "src_node", "has_tgt", "tgt_node");
        A_DELTA = add("delta", "dx", "dy");
        A_COST = add("star_cost");
        A_BUILDING = add("building", enumNames(BUILDINGS));
        A_RESOURCE = add("resource", enumNames(RESOURCES));
        A_UNIT = add("unit_type", enumNames(UNITS));
        A_LEVEL_UP = add("level_up", enumNames(LEVEL_UPS));
        A_TECH = add("tech", enumNames(TECHS));
        A_TECH_TIER = add("tech_tier");
        A_TARGET_TRIBE = add("target_tribe", slots);
        A_STARS_SENT = add("stars_sent");
        A_CAPTURE = add("capture", "city", "village");
    }

    // ---- Scheme (A) ----

    /** Size of the fixed policy head. */
    public int actionSpaceSize(int boardSize) {
        // TODO (scheme A)
        throw new UnsupportedOperationException("TODO: ActionEncoder.actionSpaceSize");
    }

    /** Index of `a` in the fixed policy head. */
    public int toIndex(Action a, GameState gs) {
        // TODO (scheme A): use a.getActionType() and the action's specific fields (look in core/actions/*).
        throw new UnsupportedOperationException("TODO: ActionEncoder.toIndex");
    }

    // ---- Scheme (B) ----

    /** Length of the vector returned by toFeatures(). */
    public int actionFeatureSize() {
        return names.size();
    }

    /** Name of every column of toFeatures(), in order. */
    public List<String> featureNames() {
        return Collections.unmodifiableList(names);
    }

    /** Features of `a`, with tribe slots relative to the active tribe. Prefer the overload with playerID. */
    public float[] toFeatures(Action a, GameState gs) {
        return toFeatures(a, gs, gs.getActiveTribeID());
    }

    /** Features of every action in `actions`, same order: row i describes actions.get(i). */
    public float[][] toFeatures(List<Action> actions, GameState gs, int playerID) {
        float[][] out = new float[actions.size()][];
        for (int i = 0; i < actions.size(); i++) out[i] = toFeatures(actions.get(i), gs, playerID);
        return out;
    }

    /**
     * Feature vector describing `a` in state `gs`. Tribe slots are relative to `playerID`, the same
     * perspective StateEncoder.encode(gs, playerID) uses, so pass the same playerID to both.
     */
    public float[] toFeatures(Action a, GameState gs, int playerID) {
        float[] f = new float[actionFeatureSize()];
        Types.ACTION type = a.getActionType();
        f[A_TYPE + type.ordinal()] = 1f;

        int size = gs.getBoard().getSize();
        int nTribes = gs.getTribes().length;
        Vector2d src = null, tgt = null;
        int cost = 0;

        if (a instanceof CityAction) {
            CityAction ca = (CityAction) a;
            src = positionOf(gs, ca.getCityId());
            tgt = ca.getTargetPos() != null ? ca.getTargetPos() : src;
            if (a instanceof Build) {
                Types.BUILDING b = ((Build) a).getBuildingType();
                if (b != null) { f[A_BUILDING + b.ordinal()] = 1f; cost = b.getCost(); }
            } else if (a instanceof ResourceGathering) {
                Types.RESOURCE r = ((ResourceGathering) a).getResource();
                if (r != null) { f[A_RESOURCE + r.ordinal()] = 1f; cost = r.getCost(); }
            } else if (a instanceof Spawn) {
                Types.UNIT u = ((Spawn) a).getUnitType();
                if (u != null) { f[A_UNIT + u.ordinal()] = 1f; cost = u.getCost(); }
            } else if (a instanceof LevelUp) {
                Types.CITY_LEVEL_UP l = ((LevelUp) a).getBonus();
                if (l != null) f[A_LEVEL_UP + l.ordinal()] = 1f;
            }
            // BURN_FOREST / CLEAR_FOREST / GROW_FOREST / DESTROY: only type + tiles.

        } else if (a instanceof UnitAction) {
            src = positionOf(gs, ((UnitAction) a).getUnitId());
            tgt = src;
            if (a instanceof Move) {
                if (((Move) a).getDestination() != null) tgt = ((Move) a).getDestination();
            } else if (a instanceof Attack) {
                Vector2d p = positionOf(gs, ((Attack) a).getTargetId());
                if (p != null) tgt = p;
            } else if (a instanceof Convert) {
                Vector2d p = positionOf(gs, ((Convert) a).getTargetId());
                if (p != null) tgt = p;
            } else if (type == Types.ACTION.CAPTURE) {
                Types.TERRAIN ct = ((core.actions.unitactions.Capture) a).getCaptureType();
                if (ct == Types.TERRAIN.CITY) f[A_CAPTURE] = 1f;
                else if (ct == Types.TERRAIN.VILLAGE) f[A_CAPTURE + 1] = 1f;
            } else if (type == Types.ACTION.UPGRADE_BOAT) {
                cost = TribesConfig.SHIP_COST;
            } else if (type == Types.ACTION.UPGRADE_SHIP) {
                cost = TribesConfig.BATTLESHIP_COST;
            }

        } else if (a instanceof BuildRoad) {
            tgt = ((BuildRoad) a).getPosition();
            cost = TribesConfig.ROAD_COST;

        } else if (a instanceof ResearchTech) {
            Types.TECHNOLOGY t = ((ResearchTech) a).getTech();
            if (t != null) {
                f[A_TECH + t.ordinal()] = 1f;
                f[A_TECH_TIER] = t.getTier() / MAX_TECH_TIER;
                Tribe actor = actingTribe(gs, ((ResearchTech) a).getTribeId());
                if (actor != null) cost = t.getCost(actor.getNumCities(), actor.getTechTree());
            }

        } else if (a instanceof DeclareWar) {
            setTargetTribe(f, ((DeclareWar) a).getTargetID(), playerID, nTribes);

        } else if (a instanceof SendStars) {
            SendStars s = (SendStars) a;
            setTargetTribe(f, s.getTargetID(), playerID, nTribes);
            f[A_STARS_SENT] = clip(s.getNumStars() / STARS_SENT_SCALE);
            cost = s.getNumStars();
        }
        // END_TURN: only the type.

        f[A_COST] = clip(cost / STAR_COST_SCALE);

        if (src != null) {
            f[A_PTR] = 1f;
            f[A_PTR + 1] = GraphState.nodeId(src.x, src.y, size);
        } else {
            f[A_PTR + 1] = -1f;
        }
        if (tgt != null) {
            f[A_PTR + 2] = 1f;
            f[A_PTR + 3] = GraphState.nodeId(tgt.x, tgt.y, size);
        } else {
            f[A_PTR + 3] = -1f;
        }
        if (src != null && tgt != null && size > 1) {
            f[A_DELTA] = (tgt.x - src.x) / (float) (size - 1);
            f[A_DELTA + 1] = (tgt.y - src.y) / (float) (size - 1);
        }
        return f;
    }

    /** Column index of the source node id (for slicing in Python). */
    public int srcNodeColumn() { return A_PTR + 1; }

    /** Column index of the target node id (for slicing in Python). */
    public int tgtNodeColumn() { return A_PTR + 3; }

    // ---------------------------------------------------------------- helpers

    private void setTargetTribe(float[] f, int targetId, int playerID, int nTribes) {
        if (targetId >= 0 && targetId < nTribes)
            f[A_TARGET_TRIBE + StateEncoder.slot(targetId, playerID, nTribes)] = 1f;
    }

    private static Vector2d positionOf(GameState gs, int actorId) {
        Actor act = gs.getActor(actorId);
        return act == null ? null : act.getPosition();
    }

    private static Tribe actingTribe(GameState gs, int tribeId) {
        if (tribeId >= 0 && tribeId < gs.getTribes().length) return gs.getTribe(tribeId);
        int active = gs.getActiveTribeID();
        return active >= 0 ? gs.getTribe(active) : null;
    }

    private static float clip(float v) {
        return v < 0f ? 0f : (v > 1f ? 1f : v);
    }

    private int add(String prefix, String... labels) {
        int off = names.size();
        if (labels.length == 0) names.add(prefix);
        else for (String l : labels) names.add(prefix + ":" + l);
        return off;
    }

    private static String[] enumNames(Enum<?>[] values) {
        String[] s = new String[values.length];
        for (int i = 0; i < values.length; i++) s[i] = values[i].name();
        return s;
    }
}
