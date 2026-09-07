#import <Foundation/Foundation.h>

@interface Widget : NSObject
- (int)render:(int)width;
@end

@implementation Widget
- (int)render:(int)width {
    return width * 2;
}
@end
