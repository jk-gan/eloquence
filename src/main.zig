const std = @import("std");
const print = std.debug.print;

const Header = extern struct { magic: u32, version: u32, dim: u32, count: u64 };

// "ELQ" + 0x01 (in hex) = 0x00514C45 (Depending on endianness, but let's just use a constant)
const MAGIC_NUMBER: u32 = 0x514C4501;

pub const DistanceMetric = enum {
    Cosine, // Normalize vectors and use dot product
    DotProduct, // Raw dot product without normalization
    Euclidean, // Use Euclidean distance
};

fn euclidean_distance(comptime dim: usize, a: @Vector(dim, f32), b: @Vector(dim, f32)) f32 {
    const diff = a - b;
    return @sqrt(@reduce(.Add, diff * diff));
}

fn euclidean_score(comptime dim: usize, a: @Vector(dim, f32), b: @Vector(dim, f32)) f32 {
    return -euclidean_distance(dim, a, b);
}

fn dot(comptime dim: usize, a: @Vector(dim, f32), b: @Vector(dim, f32)) f32 {
    return @reduce(.Add, a * b);
}

fn norm(comptime dim: usize, v: @Vector(dim, f32)) f32 {
    return @sqrt(@reduce(.Add, v * v));
}

fn normalize(comptime dim: usize, v: @Vector(dim, f32)) @Vector(dim, f32) {
    const n = norm(dim, v);
    if (n == 0.0) return v;
    return v / @as(@Vector(dim, f32), @splat(n));
}

pub fn VectorDB(comptime dim: usize, comptime metric: DistanceMetric) type {
    return struct {
        ids: std.ArrayList(u64),
        vectors: std.ArrayList(@Vector(dim, f32)),
        allocator: std.mem.Allocator,

        const Self = @This();
        const Vec = @Vector(dim, f32);

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .ids = .{}, .vectors = .{}, .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            self.ids.deinit(self.allocator);
            self.vectors.deinit(self.allocator);
        }

        pub fn add(self: *Self, id: u64, raw_vec: Vec) !void {
            const vec = if (metric == .Cosine) normalize(dim, raw_vec) else raw_vec;

            try self.vectors.append(self.allocator, vec);
            try self.ids.append(self.allocator, id);
        }

        pub const Result = struct {
            id: u64,
            score: f32,
        };

        fn compare_score(_: void, a: Result, b: Result) std.math.Order {
            return std.math.order(a.score, b.score);
        }

        pub fn search(self: *Self, raw_query: Vec, k: usize) ![]Result {
            if (self.vectors.items.len == 0) return &[0]Result{};

            const query = if (metric == .Cosine) normalize(dim, raw_query) else raw_query;
            const PriorityQueue = std.PriorityDequeue(Result, void, compare_score);

            var priority_queue = PriorityQueue.init(self.allocator, {});
            defer priority_queue.deinit();

            for (self.ids.items, self.vectors.items) |id, vec| {
                // Normally we can do cosine similarity to find similarity:
                // cosine(a, b) = dot(a, b) / (||a|| * ||b||)
                // if we normalize all vectors to unit length, we can just do:
                // cosine(a, b) = dot(a, b)  ← much faster, no sqrt or division per pair
                const score = switch (metric) {
                    .Cosine, .DotProduct => dot(dim, query, vec),
                    .Euclidean => euclidean_score(dim, query, vec),
                };

                const entry = Result{ .id = id, .score = score };

                if (priority_queue.len < k) {
                    try priority_queue.add(entry);
                } else if (score > priority_queue.peekMin().?.score) {
                    _ = priority_queue.removeMin();
                    try priority_queue.add(entry);
                }
            }

            var i = priority_queue.len;
            var results = try self.allocator.alloc(Result, i);
            while (priority_queue.removeMinOrNull()) |res| {
                i -= 1;
                results[i] = res;
            }

            return results;
        }

        pub fn save(self: *Self, path: []const u8) !void {
            var file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            const header = Header{
                .magic = MAGIC_NUMBER,
                .version = 1,
                .dim = @intCast(dim),
                .count = self.ids.items.len,
            };

            try file.writeAll(std.mem.asBytes(&header));

            const vector_bytes = std.mem.sliceAsBytes(self.vectors.items);
            try file.writeAll(vector_bytes);

            const id_bytes = std.mem.sliceAsBytes(self.ids.items);
            try file.writeAll(id_bytes);
        }

        pub fn load(allocator: std.mem.Allocator, path: []const u8) !Self {
            var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
            defer file.close();

            var header_bytes: [@sizeOf(Header)]u8 = undefined;
            const header_read = try file.readAll(&header_bytes);
            if (header_read != @sizeOf(Header)) {
                return error.UnexpectedEndOfFile;
            }

            const header: *const Header = @ptrCast(@alignCast(&header_bytes));
            if (header.magic != MAGIC_NUMBER) {
                return error.InvalidFileFormat;
            }
            if (header.dim != dim) {
                return error.DimensionMismatch;
            }

            const count: usize = @intCast(header.count);

            var vectors: std.ArrayList(Vec) = .{};
            try vectors.resize(allocator, count);

            const vector_bytes = std.mem.sliceAsBytes(vectors.items);
            const bytes_read = try file.readAll(vector_bytes);
            if (bytes_read != vector_bytes.len) {
                vectors.deinit(allocator);
                return error.UnexpectedEndOfFile;
            }

            var ids: std.ArrayList(u64) = .{};
            try ids.resize(allocator, count);

            const id_bytes = std.mem.sliceAsBytes(ids.items);
            const id_bytes_read = try file.readAll(id_bytes);
            if (id_bytes_read != id_bytes.len) {
                vectors.deinit(allocator);
                ids.deinit(allocator);
                return error.UnexpectedEndOfFile;
            }

            return Self{
                .ids = ids,
                .vectors = vectors,
                .allocator = allocator,
            };
        }
    };
}

const DB_PATH = "vectors.elq";

pub fn main() !void {
    const dim = 128;
    const DB = VectorDB(dim, .Cosine);

    var db: DB = undefined;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    if (std.fs.cwd().access(DB_PATH, .{})) |_| {
        print("Loading...\n", .{});
        db = try DB.load(allocator, DB_PATH);
    } else |_| {
        print("Generating...\n", .{});
        db = DB.init(allocator);
    }
    defer db.deinit();

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const rand = prng.random();

    print("Inserting 5,000 random vectors...\n", .{});
    for (0..5_000) |i| {
        var raw: @Vector(dim, f32) = undefined;
        for (0..dim) |j| {
            raw[j] = rand.float(f32) * 2.0 - 1.0;
        }
        try db.add(@intCast(i + 1), raw);
    }

    print("Saving...\n", .{});
    try db.save(DB_PATH);

    var query_raw: @Vector(dim, f32) = db.vectors.items[3999];

    for (0..dim) |j| {
        query_raw[j] += (rand.float(f32) * 0.001);
    }
    print("Searching top 10 neighbors...\n", .{});

    const top_10 = try db.search(query_raw, 10);
    defer allocator.free(top_10);

    for (top_10, 1..) |result, rank| {
        print("Rank {} → id: {}, score: {d:.6}\n", .{ rank, result.id, result.score });
    }
}

test "save and load roundtrip" {
    const allocator = std.testing.allocator;
    const dim = 4;
    const DB = VectorDB(dim, .Cosine);

    var db = DB.init(allocator);
    defer db.deinit();

    try db.add(1, .{ 1.0, 0.0, 0.0, 0.0 });
    try db.add(2, .{ 0.0, 1.0, 0.0, 0.0 });
    try db.add(3, .{ 0.0, 0.0, 1.0, 0.0 });

    const test_path = "test_db.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var loaded_db = try DB.load(allocator, test_path);
    defer loaded_db.deinit();

    try std.testing.expectEqual(3, loaded_db.ids.items.len);
    try std.testing.expectEqual(3, loaded_db.vectors.items.len);

    try std.testing.expectEqual(1, loaded_db.ids.items[0]);
    try std.testing.expectEqual(2, loaded_db.ids.items[1]);
    try std.testing.expectEqual(3, loaded_db.ids.items[2]);

    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(db.vectors.items[0][i], loaded_db.vectors.items[0][i], 1e-6);
        try std.testing.expectApproxEqAbs(db.vectors.items[1][i], loaded_db.vectors.items[1][i], 1e-6);
        try std.testing.expectApproxEqAbs(db.vectors.items[2][i], loaded_db.vectors.items[2][i], 1e-6);
    }
}

test "load with invalid magic returns error" {
    const allocator = std.testing.allocator;
    const dim = 4;
    const DB = VectorDB(dim, .Cosine);

    const test_path = "test_invalid.elq";

    {
        var file = try std.fs.cwd().createFile(test_path, .{});
        defer file.close();
        const invalid_header = [_]u8{0} ** @sizeOf(Header);
        try file.writeAll(&invalid_header);
    }
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const result = DB.load(allocator, test_path);
    try std.testing.expectError(error.InvalidFileFormat, result);
}

test "load with dimension mismatch returns error" {
    const allocator = std.testing.allocator;

    const DB4 = VectorDB(4, .Cosine);
    var db = DB4.init(allocator);
    defer db.deinit();

    try db.add(1, .{ 1.0, 0.0, 0.0, 0.0 });

    const test_path = "test_dim_mismatch.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const DB8 = VectorDB(8, .Cosine);
    const result = DB8.load(allocator, test_path);
    try std.testing.expectError(error.DimensionMismatch, result);
}

test "search on loaded database works correctly" {
    const allocator = std.testing.allocator;
    const dim = 4;
    const DB = VectorDB(dim, .Cosine);

    var db = DB.init(allocator);
    defer db.deinit();

    try db.add(1, .{ 1.0, 0.0, 0.0, 0.0 });
    try db.add(2, .{ 0.9, 0.1, 0.0, 0.0 });
    try db.add(3, .{ 0.0, 1.0, 0.0, 0.0 });

    const test_path = "test_search.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var loaded_db = try DB.load(allocator, test_path);
    defer loaded_db.deinit();

    const results = try loaded_db.search(.{ 1.0, 0.0, 0.0, 0.0 }, 2);
    defer allocator.free(results);

    try std.testing.expectEqual(2, results.len);
    try std.testing.expectEqual(1, results[0].id);
    try std.testing.expectEqual(2, results[1].id);
}
