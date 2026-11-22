package players.actor_critic_mcts.encoding;

import core.Diplomacy;
import core.TechnologyTree;
import core.TribesConfig;
import core.Types;
import core.actors.Actor;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Encodes a GameState as a graph (see GraphState): one node per tile, edges between 8-neighbours,
 * plus a vector of global (tribe-level) features.
 *
 * Perspective: everything is relative to `playerID`. Tribes are put in "relative slots":
 * slot 0 = playerID itself, slot k = tribe (playerID + k) mod numTribes. One network can then play any seat.
 *
 * Fog of war: the GameState an agent gets is already a partial-observation copy (unseen tiles have
 * terrain FOG, enemy stars/techs are zeroed). Fogged tiles only get their coordinates and the FOG flag;
 * everything else on them is 0, so the encoder never leaks hidden information.
 *
 * The feature layout is built once in the constructor; nodeFeatureNames(), edgeFeatureNames() and
 * globalFeatureNames() give the name of every column (handy to dump next to the training data).
 */
public class StateEncoder {

    /** Max number of tribes the layout reserves slots for. Must stay constant across a training run. */
    public static final int DEFAULT_MAX_PLAYERS = 4;

    // ---- normalisation scales (values are divided by these and clipped to [0,1]) ----
    private static final float TURN_SCALE = 50f;
    private static final float STARS_SCALE = 100f;
    private static final float SCORE_SCALE = 10000f;
    private static final float PRODUCTION_SCALE = 30f;
    private static final float COUNT_CITIES_SCALE = 15f;
    private static final float COUNT_UNITS_SCALE = 30f;
    private static final float KILLS_SCALE = 20f;
    private static final float UNIT_STAT_SCALE = 5f;    // ATK, DEF, MOV, RANGE
    private static final float UNIT_HP_SCALE = 40f;
    private static final float UNIT_KILLS_SCALE = 3f;   // 3 kills -> can become veteran
    private static final float CITY_LEVEL_SCALE = 10f;
    private static final float CITY_BOUND_SCALE = 3f;
    private static final float CITY_UNITS_SCALE = 10f;
    private static final float CITY_POINTS_SCALE = 1000f;

    // 8-neighbourhood, fixed order so edge ids are deterministic.
    private static final int[][] NEIGHBOURS = {
            {-1, -1}, {-1, 0}, {-1, 1},
            { 0, -1},          { 0, 1},
            { 1, -1}, { 1, 0}, { 1, 1}};

    private static final Types.TERRAIN[] TERRAINS = Types.TERRAIN.values();
    private static final Types.RESOURCE[] RESOURCES = Types.RESOURCE.values();
    private static final Types.BUILDING[] BUILDINGS = Types.BUILDING.values();
    private static final Types.UNIT[] UNITS = Types.UNIT.values();
    private static final Types.TURN_STATUS[] STATUSES = Types.TURN_STATUS.values();
    private static final Types.TECHNOLOGY[] TECHS = Types.TECHNOLOGY.values();
    private static final Types.TRIBE[] TRIBE_TYPES = Types.TRIBE.values();
    private static final Types.GAME_MODE[] GAME_MODES = Types.GAME_MODE.values();
    private static final Types.BUILDING[] MONUMENTS = monuments();

    private static final Map<Integer, int[][]> EDGE_INDEX_CACHE = new ConcurrentHashMap<>();

    private final int maxPlayers;

    // ---- node layout offsets ----
    private final Layout nodeLayout = new Layout();
    private final int N_POS, N_VISIBLE, N_TERRAIN, N_RESOURCE, N_BUILDING, N_HAS_BUILDING, N_ROAD, N_TRADE;
    private final int N_OWNER;              // [unowned, slot0..slotK-1, owned-by-unknown]
    private final int N_CITY;               // city centre block
    private final int N_CITY_OWNER;
    private final int N_UNIT;               // unit block
    private final int N_UNIT_OWNER, N_UNIT_TYPE, N_UNIT_STATS, N_UNIT_STATUS, N_UNIT_CARRIED;

    // ---- edge layout offsets ----
    private final Layout edgeLayout = new Layout();
    private final int E_DIAGONAL, E_ROAD_LINK, E_LAND_WATER, E_SAME_CITY, E_BORDER, E_FOG, E_UNIT_TO_EMPTY, E_DIR;

    // ---- global layout offsets ----
    private final Layout globalLayout = new Layout();
    private final int G_TURN, G_MODE, G_LEVELING_UP, G_ACTIVE, G_NUM_TRIBES;
    private final int G_OWN, G_TECHS, G_TRIBE_TYPE, G_MONUMENTS, G_OPP;
    private static final int OPP_SCALARS = 7;

    public StateEncoder() {
        this(DEFAULT_MAX_PLAYERS);
    }

    public StateEncoder(int maxPlayers) {
        this.maxPlayers = maxPlayers;
        String[] slots = slotNames(maxPlayers);

        // Node (tile) features
        N_POS = nodeLayout.add("pos", "x", "y");
        N_VISIBLE = nodeLayout.add("visible");
        N_TERRAIN = nodeLayout.add("terrain", names(TERRAINS));
        N_RESOURCE = nodeLayout.add("resource", names(RESOURCES));
        N_BUILDING = nodeLayout.add("building", names(BUILDINGS));
        N_HAS_BUILDING = nodeLayout.add("has_building");
        N_ROAD = nodeLayout.add("road");
        N_TRADE = nodeLayout.add("trade_network");
        N_OWNER = nodeLayout.add("territory_owner", concat(new String[]{"none"}, slots, new String[]{"unknown"}));
        N_CITY = nodeLayout.add("city", "is_city_centre", "level", "pop_progress", "population",
                "is_capital", "has_walls", "bound", "num_units", "points_worth");
        N_CITY_OWNER = nodeLayout.add("city_owner", slots);
        N_UNIT = nodeLayout.add("unit_present");
        N_UNIT_OWNER = nodeLayout.add("unit_owner", slots);
        N_UNIT_TYPE = nodeLayout.add("unit_type", names(UNITS));
        N_UNIT_STATS = nodeLayout.add("unit", "hp_frac", "max_hp", "atk", "def", "mov", "range",
                "kills", "veteran", "is_ranged", "is_water");
        N_UNIT_STATUS = nodeLayout.add("unit_status", names(STATUSES));
        N_UNIT_CARRIED = nodeLayout.add("unit_carried", names(UNITS));   // land unit inside a boat/ship/battleship

        // Edge features (u -> v)
        E_DIAGONAL = edgeLayout.add("diagonal");
        E_ROAD_LINK = edgeLayout.add("road_link");          // both ends on the trade network (road / city / port)
        E_LAND_WATER = edgeLayout.add("land_water");        // crossing the coast: embark / disembark
        E_SAME_CITY = edgeLayout.add("same_city");          // both ends inside the same city's borders
        E_BORDER = edgeLayout.add("border");                // territory owner differs between ends
        E_FOG = edgeLayout.add("touches_fog");              // at least one end unseen
        E_UNIT_TO_EMPTY = edgeLayout.add("unit_to_free");   // unit at u, v visible, land/water-compatible and empty
        E_DIR = edgeLayout.add("dir", "dx", "dy");

        // Global features
        G_TURN = globalLayout.add("turn");
        G_MODE = globalLayout.add("game_mode", names(GAME_MODES));
        G_LEVELING_UP = globalLayout.add("leveling_up");
        G_ACTIVE = globalLayout.add("active_tribe", slots);
        G_NUM_TRIBES = globalLayout.add("num_tribes");
        G_OWN = globalLayout.add("own", "stars", "score", "production", "num_cities", "num_units",
                "kills", "num_techs");
        G_TECHS = globalLayout.add("tech", names(TECHS));
        G_TRIBE_TYPE = globalLayout.add("tribe_type", names(TRIBE_TYPES));
        G_MONUMENTS = globalLayout.add("monument", names(MONUMENTS));
        int opp = -1;
        for (int k = 1; k < maxPlayers; k++) {
            int off = globalLayout.add("opp" + k, "present", "alive", "met", "score", "num_cities_seen",
                    "allegiance_me_to_them", "allegiance_them_to_me");
            if (k == 1) opp = off;
        }
        G_OPP = opp;
    }

    // ================================================================ public API

    /** Encode `gs` as a graph from `playerID`'s point of view. */
    public GraphState encodeGraph(GameState gs, int playerID) {
        Board board = gs.getBoard();
        int size = board.getSize();
        int nTribes = gs.getTribes().length;
        if (nTribes > maxPlayers)
            throw new IllegalArgumentException("Game has " + nTribes + " tribes but encoder reserves " + maxPlayers);

        float[][] nodes = encodeNodes(gs, board, playerID, nTribes);
        int[][] edgeIndex = edgeIndex(size);
        float[][] edges = encodeEdges(board, edgeIndex, nodes);
        float[] globals = encodeGlobals(gs, board, playerID, nTribes);
        return new GraphState(size, nodes, edgeIndex, edges, globals);
    }

    /** Flat encoding = GraphState.flatten(): node features, then edge features, then global features. */
    public float[] encode(GameState gs, int playerID) {
        return encodeGraph(gs, playerID).flatten();
    }

    /** Length of encode(). Constant for a given board size. */
    public int featureSize(int boardSize) {
        return boardSize * boardSize * nodeFeatureSize()
                + numEdges(boardSize) * edgeFeatureSize()
                + globalFeatureSize();
    }

    public int nodeFeatureSize()   { return nodeLayout.size(); }
    public int edgeFeatureSize()   { return edgeLayout.size(); }
    public int globalFeatureSize() { return globalLayout.size(); }
    public int maxPlayers()        { return maxPlayers; }

    public List<String> nodeFeatureNames()   { return nodeLayout.names(); }
    public List<String> edgeFeatureNames()   { return edgeLayout.names(); }
    public List<String> globalFeatureNames() { return globalLayout.names(); }

    public static int numEdges(int boardSize) {
        return edgeIndex(boardSize)[0].length;
    }

    /**
     * [2][E] edge list for a boardSize x boardSize grid with 8-neighbour connectivity. Cached and shared.
     * Python equivalent: for x, y in grid, for (dx, dy) in NEIGHBOURS order, if in bounds -> (id(x,y), id(x+dx,y+dy)).
     */
    public static int[][] edgeIndex(int boardSize) {
        return EDGE_INDEX_CACHE.computeIfAbsent(boardSize, StateEncoder::buildEdgeIndex);
    }

    // ================================================================ nodes

    float[][] encodeNodes(GameState gs, Board board, int me, int nTribes) {
        int size = board.getSize();
        float[][] nodes = new float[size * size][nodeFeatureSize()];
        for (int x = 0; x < size; x++)
            for (int y = 0; y < size; y++)
                encodeTile(gs, board, me, nTribes, x, y, nodes[GraphState.nodeId(x, y, size)]);
        return nodes;
    }

    private void encodeTile(GameState gs, Board board, int me, int nTribes, int x, int y, float[] f) {
        int size = board.getSize();
        f[N_POS] = size > 1 ? x / (float) (size - 1) : 0f;
        f[N_POS + 1] = size > 1 ? y / (float) (size - 1) : 0f;

        Types.TERRAIN terrain = board.getTerrainAt(x, y);
        if (terrain == null || terrain == Types.TERRAIN.FOG) {
            f[N_TERRAIN + Types.TERRAIN.FOG.ordinal()] = 1f;
            return;   // nothing else is observable on a fogged tile
        }
        f[N_VISIBLE] = 1f;
        f[N_TERRAIN + terrain.ordinal()] = 1f;

        Types.RESOURCE res = board.getResourceAt(x, y);   // already masked by tech in partial-obs copies
        if (res != null) f[N_RESOURCE + res.ordinal()] = 1f;

        Types.BUILDING bld = board.getBuildingAt(x, y);
        if (bld != null) {
            f[N_BUILDING + bld.ordinal()] = 1f;
            f[N_HAS_BUILDING] = 1f;
        }

        if (board.isRoad(x, y)) f[N_ROAD] = 1f;
        if (board.checkTradeNetwork(x, y)) f[N_TRADE] = 1f;

        // Territory owner
        int cityId = board.getCityIdAt(x, y);
        if (cityId == -1) {
            f[N_OWNER] = 1f;
        } else {
            Actor a = board.getActor(cityId);
            if (a instanceof City) f[N_OWNER + 1 + slot(a.getTribeId(), me, nTribes)] = 1f;
            else f[N_OWNER + 1 + maxPlayers] = 1f;   // owned, but the owning city isn't visible
        }

        // City centre
        if (terrain == Types.TERRAIN.CITY && cityId != -1 && board.getActor(cityId) instanceof City) {
            City c = (City) board.getActor(cityId);
            int need = Math.max(1, c.getPopulation_need());
            f[N_CITY] = 1f;
            f[N_CITY + 1] = clip(c.getLevel() / CITY_LEVEL_SCALE);
            f[N_CITY + 2] = clip(c.getPopulation() / (float) need);
            f[N_CITY + 3] = clip(c.getPopulation() / CITY_LEVEL_SCALE);
            f[N_CITY + 4] = c.isCapital() ? 1f : 0f;
            f[N_CITY + 5] = c.hasWalls() ? 1f : 0f;
            f[N_CITY + 6] = clip(c.getBound() / CITY_BOUND_SCALE);
            f[N_CITY + 7] = clip(c.getNumUnits() / CITY_UNITS_SCALE);   // 0 for enemy cities (hidden)
            f[N_CITY + 8] = clip(c.getPointsWorth() / CITY_POINTS_SCALE);
            f[N_CITY_OWNER + slot(c.getTribeId(), me, nTribes)] = 1f;
        }

        // Unit
        Unit u = board.getUnitAt(x, y);
        if (u != null) {
            Types.UNIT type = u.getType();
            f[N_UNIT] = 1f;
            f[N_UNIT_OWNER + slot(u.getTribeId(), me, nTribes)] = 1f;
            f[N_UNIT_TYPE + type.ordinal()] = 1f;
            int maxHp = Math.max(1, u.getMaxHP());
            f[N_UNIT_STATS] = clip(u.getCurrentHP() / (float) maxHp);
            f[N_UNIT_STATS + 1] = clip(maxHp / UNIT_HP_SCALE);
            f[N_UNIT_STATS + 2] = clip(u.ATK / UNIT_STAT_SCALE);
            f[N_UNIT_STATS + 3] = clip(u.DEF / UNIT_STAT_SCALE);
            f[N_UNIT_STATS + 4] = clip(u.MOV / UNIT_STAT_SCALE);
            f[N_UNIT_STATS + 5] = clip(u.RANGE / UNIT_STAT_SCALE);
            f[N_UNIT_STATS + 6] = clip(u.getKills() / UNIT_KILLS_SCALE);
            f[N_UNIT_STATS + 7] = u.isVeteran() ? 1f : 0f;
            f[N_UNIT_STATS + 8] = type.isRanged() ? 1f : 0f;
            f[N_UNIT_STATS + 9] = type.isWaterUnit() ? 1f : 0f;
            if (u.getStatus() != null) f[N_UNIT_STATUS + u.getStatus().ordinal()] = 1f;
            if (type == Types.UNIT.BOAT || type == Types.UNIT.SHIP || type == Types.UNIT.BATTLESHIP) {
                Types.UNIT carried = board.getBaseLandUnit(u);
                if (carried != null) f[N_UNIT_CARRIED + carried.ordinal()] = 1f;
            }
        }
    }

    // ================================================================ edges

    float[][] encodeEdges(Board board, int[][] edgeIndex, float[][] nodes) {
        int size = board.getSize();
        int nE = edgeIndex[0].length;
        float[][] edges = new float[nE][edgeFeatureSize()];
        for (int e = 0; e < nE; e++) {
            int s = edgeIndex[0][e], t = edgeIndex[1][e];
            int sx = s / size, sy = s % size, tx = t / size, ty = t % size;
            int dx = tx - sx, dy = ty - sy;
            float[] f = edges[e];
            float[] ns = nodes[s], nt = nodes[t];

            f[E_DIAGONAL] = (dx != 0 && dy != 0) ? 1f : 0f;
            f[E_DIR] = dx;
            f[E_DIR + 1] = dy;

            boolean sVis = ns[N_VISIBLE] > 0, tVis = nt[N_VISIBLE] > 0;
            if (!sVis || !tVis) {
                f[E_FOG] = 1f;
                continue;
            }

            if (board.checkTradeNetwork(sx, sy) && board.checkTradeNetwork(tx, ty)) f[E_ROAD_LINK] = 1f;

            boolean sWater = board.getTerrainAt(sx, sy).isWater(), tWater = board.getTerrainAt(tx, ty).isWater();
            if (sWater != tWater) f[E_LAND_WATER] = 1f;

            int sCity = board.getCityIdAt(sx, sy), tCity = board.getCityIdAt(tx, ty);
            if (sCity != -1 && sCity == tCity) f[E_SAME_CITY] = 1f;
            if (ownerColumn(ns) != ownerColumn(nt)) f[E_BORDER] = 1f;

            if (ns[N_UNIT] > 0 && nt[N_UNIT] == 0) {
                boolean unitOnWater = ns[N_UNIT_STATS + 9] > 0;
                if (unitOnWater == tWater || board.getTerrainAt(tx, ty) == Types.TERRAIN.CITY) f[E_UNIT_TO_EMPTY] = 1f;
            }
        }
        return edges;
    }

    private int ownerColumn(float[] node) {
        for (int i = 0; i < maxPlayers + 2; i++)
            if (node[N_OWNER + i] > 0) return i;
        return -1;
    }

    // ================================================================ globals

    float[] encodeGlobals(GameState gs, Board board, int me, int nTribes) {
        float[] g = new float[globalFeatureSize()];
        Tribe own = gs.getTribe(me);

        g[G_TURN] = clip(gs.getTick() / TURN_SCALE);
        if (gs.getGameMode() != null) g[G_MODE + gs.getGameMode().ordinal()] = 1f;
        g[G_LEVELING_UP] = gs.isLevelingUp() ? 1f : 0f;
        int active = gs.getActiveTribeID();
        if (active >= 0) g[G_ACTIVE + slot(active, me, nTribes)] = 1f;
        g[G_NUM_TRIBES] = nTribes / (float) maxPlayers;

        TechnologyTree tree = own.getTechTree();
        g[G_OWN] = clip(own.getStars() / STARS_SCALE);
        g[G_OWN + 1] = clip(own.getScore() / SCORE_SCALE);
        g[G_OWN + 2] = clip(safeProduction(gs, me) / PRODUCTION_SCALE);
        g[G_OWN + 3] = clip(own.getNumCities() / COUNT_CITIES_SCALE);
        g[G_OWN + 4] = clip(safeUnitCount(gs, me) / COUNT_UNITS_SCALE);
        g[G_OWN + 5] = clip(own.getnKills() / KILLS_SCALE);
        g[G_OWN + 6] = tree.getNumResearched() / (float) TECHS.length;

        for (Types.TECHNOLOGY t : TECHS)
            if (tree.isResearched(t)) g[G_TECHS + t.ordinal()] = 1f;

        if (own.getType() != null) g[G_TRIBE_TYPE + own.getType().ordinal()] = 1f;

        HashMap<Types.BUILDING, Types.BUILDING.MONUMENT_STATUS> mons = own.getMonuments();
        for (int i = 0; i < MONUMENTS.length; i++) {
            Types.BUILDING.MONUMENT_STATUS st = mons == null ? null : mons.get(MONUMENTS[i]);
            if (st != null) g[G_MONUMENTS + i] = st.getKey() / 2f;   // 0 unavailable, .5 available, 1 built
        }

        ArrayList<Integer> met = own.getTribesMet();
        Diplomacy dip = board.getDiplomacy();
        int[][] alleg = dip == null ? null : dip.getAllegianceStatus();
        for (int k = 1; k < maxPlayers; k++) {
            int off = G_OPP + (k - 1) * OPP_SCALARS;
            if (k >= nTribes) continue;
            int opp = (me + k) % nTribes;
            Tribe t = gs.getTribe(opp);
            g[off] = 1f;
            g[off + 1] = t.getWinner() == Types.RESULT.LOSS ? 0f : 1f;
            g[off + 2] = met != null && met.contains(opp) ? 1f : 0f;
            g[off + 3] = clip(t.getScore() / SCORE_SCALE);
            g[off + 4] = clip(t.getNumCities() / COUNT_CITIES_SCALE);   // only cities we can see
            if (alleg != null) {
                g[off + 5] = alleg[me][opp] / (float) TribesConfig.ALLEGIANCE_MAX;   // in [-1, 1]
                g[off + 6] = alleg[opp][me] / (float) TribesConfig.ALLEGIANCE_MAX;
            }
        }
        return g;
    }

    private static int safeProduction(GameState gs, int me) {
        try { return gs.getTribeProduction(me); } catch (RuntimeException e) { return 0; }
    }

    private static int safeUnitCount(GameState gs, int me) {
        try { return gs.getUnits(me).size(); } catch (RuntimeException e) { return 0; }
    }

    // ================================================================ helpers

    /** Relative slot of `tribeId` seen from `me`: 0 = me, 1 = next tribe, ... */
    static int slot(int tribeId, int me, int nTribes) {
        return ((tribeId - me) % nTribes + nTribes) % nTribes;
    }

    private static float clip(float v) {
        return v < 0f ? 0f : (v > 1f ? 1f : v);
    }

    private static int[][] buildEdgeIndex(int size) {
        ArrayList<int[]> list = new ArrayList<>();
        for (int x = 0; x < size; x++)
            for (int y = 0; y < size; y++)
                for (int[] d : NEIGHBOURS) {
                    int nx = x + d[0], ny = y + d[1];
                    if (nx >= 0 && ny >= 0 && nx < size && ny < size)
                        list.add(new int[]{GraphState.nodeId(x, y, size), GraphState.nodeId(nx, ny, size)});
                }
        int[][] ei = new int[2][list.size()];
        for (int e = 0; e < list.size(); e++) {
            ei[0][e] = list.get(e)[0];
            ei[1][e] = list.get(e)[1];
        }
        return ei;
    }

    private static Types.BUILDING[] monuments() {
        ArrayList<Types.BUILDING> m = new ArrayList<>();
        for (Types.BUILDING b : Types.BUILDING.values()) if (b.isMonument()) m.add(b);
        return m.toArray(new Types.BUILDING[0]);
    }

    private static String[] slotNames(int maxPlayers) {
        String[] s = new String[maxPlayers];
        s[0] = "self";
        for (int k = 1; k < maxPlayers; k++) s[k] = "opp" + k;
        return s;
    }

    private static String[] names(Enum<?>[] values) {
        String[] s = new String[values.length];
        for (int i = 0; i < values.length; i++) s[i] = values[i].name();
        return s;
    }

    private static String[] concat(String[]... parts) {
        ArrayList<String> out = new ArrayList<>();
        for (String[] p : parts) Collections.addAll(out, p);
        return out.toArray(new String[0]);
    }

    /** Ordered list of named feature columns; add() returns the offset of the block it appended. */
    private static final class Layout {
        private final ArrayList<String> names = new ArrayList<>();

        int add(String prefix, String... labels) {
            int off = names.size();
            if (labels.length == 0) names.add(prefix);
            else for (String l : labels) names.add(prefix + ":" + l);
            return off;
        }

        int size() { return names.size(); }

        List<String> names() { return Collections.unmodifiableList(names); }
    }
}
